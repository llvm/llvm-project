//===-- Variables.h -----------------------------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_TOOLS_LLDB_DAP_VARIABLES_H
#define LLDB_TOOLS_LLDB_DAP_VARIABLES_H

#include "DAPForward.h"
#include "DAPLog.h"
#include "Protocol/DAPTypes.h"
#include "Protocol/ProtocolRequests.h"
#include "Protocol/ProtocolTypes.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBValue.h"
#include "lldb/API/SBValueList.h"
#include "llvm/Support/ErrorHandling.h"

namespace lldb_dap {
struct VariableReferenceStorage;

enum ScopeKind : unsigned {
  eScopeKindLocals,
  eScopeKindGlobals,
  eScopeKindRegisters
};

/// Creates a `protocol::Scope` struct.
///
/// \param[in] kind
///     The kind of scope to create
///
/// \param[in] variablesReference
///     The value to place into the "variablesReference" key
///
/// \param[in] expensive
///     The value to place into the "expensive" key
///
/// \return
///     A `protocol::Scope`
protocol::Scope CreateScope(ScopeKind kind, var_ref_t variablesReference,
                            bool expensive);

/// An Interface to get or find specific variables by name.
class VariableStore {
public:
  explicit VariableStore() = default;
  virtual ~VariableStore() = default;

  virtual llvm::Expected<std::vector<protocol::Variable>>
  GetVariables(VariableReferenceStorage &storage,
               const protocol::Configuration &config,
               const protocol::VariablesArguments &args) = 0;
  virtual lldb::SBValue FindVariable(llvm::StringRef name) = 0;
  virtual lldb::SBValue GetVariable() const = 0;

  // Not copyable.
  VariableStore(const VariableStore &) = delete;
  VariableStore &operator=(const VariableStore &) = delete;
  VariableStore(VariableStore &&) = default;
  VariableStore &operator=(VariableStore &&) = default;
};

struct VariableReferenceStorage {
  explicit VariableReferenceStorage(Log &log) : log(log) {}
  /// \return a new variableReference.
  /// Specify is_permanent as true for variable that should persist entire
  /// debug session.
  var_ref_t CreateVariableReference(bool is_permanent);

  /// \return the expandable variable corresponding with variableReference
  /// value of \p value.
  /// If \p var_ref is invalid an empty SBValue is returned.
  lldb::SBValue GetVariable(var_ref_t var_ref);

  /// Insert a new \p variable.
  /// \return variableReference assigned to this expandable variable.
  var_ref_t Insert(const lldb::SBValue &variable, bool is_permanent);

  /// Insert a value list. Used to store references to lldb repl command
  /// outputs.
  var_ref_t Insert(const lldb::SBValueList &values);

  /// Insert a new frame into temporary storage.
  std::vector<protocol::Scope> Insert(const lldb::SBFrame &frame);

  lldb::SBValue FindVariable(var_ref_t var_ref, llvm::StringRef name);

  void Clear() { m_temporary_kind_pool.Clear(); }

  VariableStore *GetVariableStore(var_ref_t var_ref);
  Log &log;

private:
  /// Template class for managing pools of variable stores.
  /// All references created starts from zero with the Reference kind mask
  /// applied, the mask is then removed when fetching a variable store
  ///
  /// \tparam ReferenceKind
  ///     The reference kind created in this pool
  template <protocol::ReferenceKind Kind> class ReferenceKindPool {

  public:
    explicit ReferenceKindPool() = default;

    /// Resets the count to zero and clears the pool,
    /// disabled for permanent reference kind.
    template <protocol::ReferenceKind LHS = Kind,
              protocol::ReferenceKind RHS = protocol::eReferenceKindPermanent>
    std::enable_if_t<LHS != RHS, void> Clear() {
      reference_count = 0;
      m_pool.clear();
    }

    VariableStore *GetVariableStore(var_ref_t var_ref) {
      const uint32_t raw_ref = var_ref.Reference();

      if (raw_ref != 0 && raw_ref <= m_pool.size())
        return m_pool[raw_ref - 1].get();
      return nullptr;
    }

    template <typename T, typename... Args> var_ref_t Add(Args &&...args) {
      assert(reference_count == m_pool.size() &&
             "Current reference_count must be the size of the pool");

      if (LLVM_UNLIKELY(reference_count >=
                        var_ref_t::k_max_variables_references)) {
        // We cannot add new variables to the pool;
        return var_ref_t(var_ref_t::k_invalid_var_ref);
      }

      m_pool.emplace_back(std::make_unique<T>(std::forward<Args>(args)...));
      const uint32_t raw_ref = NextRawReference();
      return var_ref_t(raw_ref, Kind);
    }

    [[nodiscard]] size_t Size() const { return m_pool.size(); }

    // Non copyable and non movable.
    ReferenceKindPool(const ReferenceKindPool &) = delete;
    ReferenceKindPool &operator=(const ReferenceKindPool &) = delete;
    ReferenceKindPool(ReferenceKindPool &&) = delete;
    ReferenceKindPool &operator=(ReferenceKindPool &&) = delete;
    ~ReferenceKindPool() = default;

  private:
    uint32_t NextRawReference() {
      reference_count++;
      return reference_count;
    }

    uint32_t reference_count = 0;
    std::vector<std::unique_ptr<VariableStore>> m_pool;
  };

  /// Variables that are alive in this stop state.
  /// Will be cleared when debuggee resumes.
  ReferenceKindPool<protocol::eReferenceKindTemporary> m_temporary_kind_pool;
  /// Variables that persist across entire debug session.
  /// These are the variables evaluated from debug console REPL.
  ReferenceKindPool<protocol::eReferenceKindPermanent> m_permanent_kind_pool;
};

} // namespace lldb_dap

#endif
