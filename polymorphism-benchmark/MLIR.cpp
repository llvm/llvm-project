//===- MLIR.cpp - MLIR-style extensible polymorphism ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic Operation objects refer to a registered OperationName, which holds
// the op's TypeID, a sorted InterfaceMap of Concept implementations, and a
// virtual hasTrait(). Ops are thin Op<ConcreteOp, Traits...> wrappers around
// an Operation*. Abstract classes are modelled as
//
//  * a trait (IsUnaryOp, IsBinaryOp) that provides shared member functions,
//    such as the getCost() used by all binary ops except DivOp, and
//  * an interface (BinaryOpInterface) for generic access.
//
// Dispatch and type tests use the idioms found in MLIR code: dyn_cast to an
// interface, TypeSwitch, hasTrait<>, and isa<> over a list of ops.
//
//===----------------------------------------------------------------------===//

#include "Common.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

namespace polybench {
namespace POLY_NS(mlirstyle) {

/// Unique identifier of a C++ type: the address of a per-type static object.
class TypeID {
public:
  TypeID() = default;

  template <typename T> static TypeID get() {
    static const char Storage = 0;
    return TypeID(&Storage);
  }

  const void *getAsOpaquePointer() const { return Storage; }

  friend bool operator==(TypeID L, TypeID R) { return L.Storage == R.Storage; }
  friend bool operator!=(TypeID L, TypeID R) { return L.Storage != R.Storage; }

private:
  explicit TypeID(const void *Storage) : Storage(Storage) {}

  const void *Storage = nullptr;
};

/// Gives traits, which are class templates, a TypeID.
template <template <typename> class Trait> struct TraitTag {};

/// Maps interface TypeIDs to concept implementations; kept sorted by TypeID
/// and searched with lower_bound, like mlir::detail::InterfaceMap.
class InterfaceMap {
public:
  template <typename ModelT> void insertModel(TypeID InterfaceID) {
    auto *Model = new ModelT();
    Owned.emplace_back(Model, [](void *M) { delete static_cast<ModelT *>(M); });
    insert(InterfaceID, Model);
  }

  void *lookup(TypeID ID) const {
    const auto *Begin = Interfaces.data();
    const auto *End = Begin + Interfaces.size();
    const auto *It = std::lower_bound(
        Begin, End, ID, [](const std::pair<TypeID, void *> &Entry, TypeID ID) {
          return compare(Entry.first, ID);
        });
    return (It != End && It->first == ID) ? It->second : nullptr;
  }

private:
  static bool compare(TypeID L, TypeID R) {
    return L.getAsOpaquePointer() < R.getAsOpaquePointer();
  }

  void insert(TypeID ID, void *Concept) {
    auto It =
        std::lower_bound(Interfaces.begin(), Interfaces.end(), ID,
                         [](const std::pair<TypeID, void *> &Entry, TypeID ID) {
                           return compare(Entry.first, ID);
                         });
    Interfaces.insert(It, {ID, Concept});
  }

  std::vector<std::pair<TypeID, void *>> Interfaces;
  std::vector<std::unique_ptr<void, void (*)(void *)>> Owned;
};

/// Information about an operation kind (OperationName::Impl).
class OperationName {
public:
  virtual ~OperationName() = default;

  bool isRegistered() const { return ID != TypeID::get<void>(); }
  TypeID getTypeID() const { return ID; }
  const InterfaceMap &getInterfaceMap() const { return Interfaces; }

  template <template <typename> class Trait> bool hasTrait() const {
    return hasTrait(TypeID::get<TraitTag<Trait>>());
  }
  virtual bool hasTrait(TypeID TraitID) const = 0;

protected:
  explicit OperationName(TypeID ID) : ID(ID) {}

  InterfaceMap Interfaces;

private:
  TypeID ID;
};

/// Registered information of a concrete op
/// (RegisteredOperationName::Model<ConcreteOp>).
template <typename ConcreteOp>
class RegisteredOperationModel final : public OperationName {
public:
  RegisteredOperationModel() : OperationName(TypeID::get<ConcreteOp>()) {
    ConcreteOp::attachInterfaces(Interfaces);
  }

  bool hasTrait(TypeID TraitID) const override {
    return ConcreteOp::hasTraitID(TraitID);
  }
};

/// Generic operation storage. Op-specific data lives here rather than in the
/// Op wrapper classes; A and B stand in for operands/properties.
class Operation {
public:
  Operation(const OperationName *Name, int A, int B) : Name(Name), A(A), B(B) {}

  const OperationName *getName() const { return Name; }

  const OperationName *getRegisteredInfo() const {
    return Name->isRegistered() ? Name : nullptr;
  }

  template <template <typename> class Trait> bool hasTrait() const {
    return Name->hasTrait<Trait>();
  }

  int getA() const { return A; }
  int getB() const { return B; }

private:
  const OperationName *Name;
  int A, B;
};

/// Base of Op wrappers and interfaces: a nullable pointer to an Operation.
class OpState {
public:
  explicit OpState(Operation *Op) : State(Op) {}
  Operation *getOperation() const { return State; }
  explicit operator bool() const { return State != nullptr; }

private:
  Operation *State;
};

template <typename First, typename... Rest> bool isa(Operation *Op) {
  return First::classof(Op) || (Rest::classof(Op) || ...);
}

/// Like mlir's CastInfo for Operation*: check classof(), then construct.
template <typename To> To dyn_cast(Operation *Op) {
  return isa<To>(Op) ? To(Op) : To(nullptr);
}

/// Base for interfaces: the constructor resolves the concept of the
/// operation's registered model.
template <typename ConcreteType, typename Traits>
class OpInterface : public OpState {
public:
  using Concept = typename Traits::Concept;
  template <typename T> using Model = typename Traits::template Model<T>;

  explicit OpInterface(Operation *Op = nullptr)
      : OpState(Op), Impl(Op ? getInterfaceFor(Op) : nullptr) {}

  static TypeID getInterfaceID() { return TypeID::get<ConcreteType>(); }

  POLY_METHOD static bool classof(Operation *Op) {
    return getInterfaceFor(Op) != nullptr;
  }

protected:
  const Concept *getImpl() const { return Impl; }

private:
  static const Concept *getInterfaceFor(Operation *Op) {
    if (const OperationName *Info = Op->getRegisteredInfo())
      return static_cast<const Concept *>(
          Info->getInterfaceMap().lookup(getInterfaceID()));
    return nullptr;
  }

  const Concept *Impl;
};

/// Detects op traits that are interface traits, i.e. that bring a model.
template <typename T, typename = void>
struct IsInterfaceTrait : std::false_type {};
template <typename T>
struct IsInterfaceTrait<
    T, std::void_t<decltype(T::getInterfaceID()), typename T::ModelT>>
    : std::true_type {};

template <typename ConcreteType, template <typename> class... Traits>
class Op : public OpState, public Traits<ConcreteType>... {
public:
  explicit Op(Operation *O = nullptr) : OpState(O) {}

  POLY_METHOD static bool classof(Operation *O) {
    if (const OperationName *Info = O->getRegisteredInfo())
      return TypeID::get<ConcreteType>() == Info->getTypeID();
    return false;
  }

  static void attachInterfaces(InterfaceMap &Map) {
    (attachIfInterface<Traits<ConcreteType>>(Map), ...);
  }

  static bool hasTraitID(TypeID TraitID) {
    const TypeID TraitIDs[] = {TypeID::get<TraitTag<Traits>>()...};
    for (TypeID ID : TraitIDs)
      if (ID == TraitID)
        return true;
    return false;
  }

private:
  template <typename TraitT> static void attachIfInterface(InterfaceMap &Map) {
    if constexpr (IsInterfaceTrait<TraitT>::value)
      Map.insertModel<typename TraitT::ModelT>(TraitT::getInterfaceID());
  }
};

/// CRTP base of traits (OpTrait::TraitBase). Parameterized by the trait so
/// that each trait has its own base subobject.
template <typename ConcreteOp, template <typename> class TraitType>
class TraitBase {
protected:
  Operation *getOp() const {
    return static_cast<const ConcreteOp *>(this)->getOperation();
  }
};

template <typename ConcreteOp> class ZeroOperands {};

template <typename ConcreteOp>
class IsUnaryOp : public TraitBase<ConcreteOp, IsUnaryOp> {
public:
  int getOperand() const { return this->getOp()->getA(); }
};

template <typename ConcreteOp>
class IsBinaryOp : public TraitBase<ConcreteOp, IsBinaryOp> {
public:
  int getLHS() const { return this->getOp()->getA(); }
  int getRHS() const { return this->getOp()->getB(); }
  POLY_METHOD std::int64_t getCost() const { return getLHS() + getRHS() + 1; }
};

/// The dispatched interface, as ODS would generate it.
struct CostInterfaceTraits {
  struct Concept {
    std::int64_t (*getCost)(const Concept *Impl, Operation *Op);
  };
  template <typename ConcreteOp> struct Model : Concept {
    Model() : Concept{&getCostImpl} {}
    static std::int64_t getCostImpl(const Concept *, Operation *Op) {
      return ConcreteOp(Op).getCost();
    }
  };
};

class CostInterface : public OpInterface<CostInterface, CostInterfaceTraits> {
public:
  using OpInterface::OpInterface;

  std::int64_t getCost() const {
    return getImpl()->getCost(getImpl(), getOperation());
  }

  template <typename ConcreteOp> struct Trait {
    static TypeID getInterfaceID() { return CostInterface::getInterfaceID(); }
    using ModelT = CostInterfaceTraits::Model<ConcreteOp>;
  };
};

/// Generic access to binary operations.
struct BinaryOpInterfaceTraits {
  struct Concept {
    int (*getLHS)(const Concept *Impl, Operation *Op);
    int (*getRHS)(const Concept *Impl, Operation *Op);
  };
  template <typename ConcreteOp> struct Model : Concept {
    Model() : Concept{&getLHSImpl, &getRHSImpl} {}
    static int getLHSImpl(const Concept *, Operation *Op) {
      return ConcreteOp(Op).getLHS();
    }
    static int getRHSImpl(const Concept *, Operation *Op) {
      return ConcreteOp(Op).getRHS();
    }
  };
};

class BinaryOpInterface
    : public OpInterface<BinaryOpInterface, BinaryOpInterfaceTraits> {
public:
  using OpInterface::OpInterface;

  int getLHS() const { return getImpl()->getLHS(getImpl(), getOperation()); }
  int getRHS() const { return getImpl()->getRHS(getImpl(), getOperation()); }

  template <typename ConcreteOp> struct Trait {
    static TypeID getInterfaceID() {
      return BinaryOpInterface::getInterfaceID();
    }
    using ModelT = BinaryOpInterfaceTraits::Model<ConcreteOp>;
  };
};

/// Additional interfaces so that the InterfaceMap lookup has to search.
template <int N> struct DummyInterfaceTraits {
  struct Concept {
    int (*dummy)(const Concept *Impl, Operation *Op);
  };
  template <typename ConcreteOp> struct Model : Concept {
    Model() : Concept{&dummyImpl} {}
    static int dummyImpl(const Concept *, Operation *) { return N; }
  };
};

template <int N>
class DummyInterface
    : public OpInterface<DummyInterface<N>, DummyInterfaceTraits<N>> {
public:
  using OpInterface<DummyInterface<N>, DummyInterfaceTraits<N>>::OpInterface;

  template <typename ConcreteOp> struct Trait {
    static TypeID getInterfaceID() { return DummyInterface::getInterfaceID(); }
    using ModelT = typename DummyInterfaceTraits<N>::template Model<ConcreteOp>;
  };
};

#define COMMON_OP_TRAITS                                                       \
  DummyInterface<0>::Trait, CostInterface::Trait, DummyInterface<1>::Trait,    \
      DummyInterface<2>::Trait
#define BINARY_OP_TRAITS IsBinaryOp, BinaryOpInterface::Trait, COMMON_OP_TRAITS

class ConstOp : public Op<ConstOp, ZeroOperands, COMMON_OP_TRAITS> {
public:
  using Op::Op;
  int getValue() const { return getOperation()->getA(); }
  POLY_METHOD std::int64_t getCost() const { return 1; }
};

class NegOp : public Op<NegOp, IsUnaryOp, COMMON_OP_TRAITS> {
public:
  using Op::Op;
  POLY_METHOD std::int64_t getCost() const { return getOperand() + 1; }
};

class NotOp : public Op<NotOp, IsUnaryOp, COMMON_OP_TRAITS> {
public:
  using Op::Op;
  POLY_METHOD std::int64_t getCost() const { return getOperand() + 2; }
};

class AddOp : public Op<AddOp, BINARY_OP_TRAITS> {
public:
  using Op::Op;
};

class SubOp : public Op<SubOp, BINARY_OP_TRAITS> {
public:
  using Op::Op;
};

class MulOp : public Op<MulOp, BINARY_OP_TRAITS> {
public:
  using Op::Op;
};

class DivOp : public Op<DivOp, BINARY_OP_TRAITS> {
public:
  using Op::Op;
  POLY_METHOD std::int64_t getCost() const { return getLHS() + getRHS() + 20; }
};

class CallOp : public Op<CallOp, COMMON_OP_TRAITS> {
public:
  using Op::Op;
  int getNumArgs() const { return getOperation()->getA(); }
  int getCallee() const { return getOperation()->getB(); }
  POLY_METHOD std::int64_t getCost() const { return getNumArgs() * 4 + 10; }
};

#undef BINARY_OP_TRAITS
#undef COMMON_OP_TRAITS

/// Minimal llvm::TypeSwitch.
template <typename ResultT> class TypeSwitch {
public:
  explicit TypeSwitch(Operation *Value) : Value(Value) {}

  template <typename CaseT, typename CallableT>
  TypeSwitch &Case(CallableT &&Fn) {
    if (Result)
      return *this;
    if (auto CaseValue = dyn_cast<CaseT>(Value))
      Result.emplace(Fn(CaseValue));
    return *this;
  }

  template <typename CallableT> ResultT Default(CallableT &&Fn) {
    if (Result)
      return std::move(*Result);
    return Fn(Value);
  }

private:
  Operation *Value;
  std::optional<ResultT> Result;
};

/// Owns the registered operation names (the MLIRContext's role).
class Context {
public:
  template <typename OpT> const OperationName *getOrRegister() {
    TypeID ID = TypeID::get<OpT>();
    for (const auto &Name : Names)
      if (Name->getTypeID() == ID)
        return Name.get();
    Names.push_back(std::make_unique<RegisteredOperationModel<OpT>>());
    return Names.back().get();
  }

private:
  std::vector<std::unique_ptr<OperationName>> Names;
};

struct Nodes {
  Context Ctx;
  std::vector<std::unique_ptr<Operation>> Owner;
  std::vector<Operation *> Ptrs;

  explicit Nodes(Pattern P) {
    for (const NodeSpec &S : makeSpecs(P)) {
      Owner.push_back(std::make_unique<Operation>(getName(S.K), S.A, S.B));
      Ptrs.push_back(Owner.back().get());
    }
  }

private:
  const OperationName *getName(Kind K) {
    switch (K) {
    case Kind::Const:
      return Ctx.getOrRegister<ConstOp>();
    case Kind::Neg:
      return Ctx.getOrRegister<NegOp>();
    case Kind::Not:
      return Ctx.getOrRegister<NotOp>();
    case Kind::Add:
      return Ctx.getOrRegister<AddOp>();
    case Kind::Sub:
      return Ctx.getOrRegister<SubOp>();
    case Kind::Mul:
      return Ctx.getOrRegister<MulOp>();
    case Kind::Div:
      return Ctx.getOrRegister<DivOp>();
    case Kind::Call:
      return Ctx.getOrRegister<CallOp>();
    }
    return nullptr;
  }
};

/// if (auto Iface = dyn_cast<CostInterface>(Op)) Iface.getCost();
void BM_Dispatch_Interface(benchmark::State &State, Pattern P) {
  Nodes N(P);
  runBenchmark(State, P, N.Ptrs, [](Operation *Op) -> std::int64_t {
    if (auto Iface = dyn_cast<CostInterface>(Op))
      return Iface.getCost();
    return 0;
  });
}

/// Interface objects resolved ahead of time; measures only the indirect call
/// through the Concept.
void BM_Dispatch_InterfaceCached(benchmark::State &State, Pattern P) {
  Nodes N(P);
  std::vector<CostInterface> Ifaces;
  for (Operation *Op : N.Ptrs)
    Ifaces.push_back(CostInterface(Op));
  runBenchmark(State, P, Ifaces,
               [](const CostInterface &Iface) -> std::int64_t {
                 return Iface.getCost();
               });
}

void BM_Dispatch_TypeSwitch(benchmark::State &State, Pattern P) {
  Nodes N(P);
  runBenchmark(State, P, N.Ptrs, [](Operation *Op) -> std::int64_t {
    return TypeSwitch<std::int64_t>(Op)
        .Case<ConstOp>([](ConstOp C) { return C.getCost(); })
        .Case<NegOp>([](NegOp C) { return C.getCost(); })
        .Case<NotOp>([](NotOp C) { return C.getCost(); })
        .Case<AddOp>([](AddOp C) { return C.getCost(); })
        .Case<SubOp>([](SubOp C) { return C.getCost(); })
        .Case<MulOp>([](MulOp C) { return C.getCost(); })
        .Case<DivOp>([](DivOp C) { return C.getCost(); })
        .Case<CallOp>([](CallOp C) { return C.getCost(); })
        .Default([](Operation *) -> std::int64_t { return 0; });
  });
}

/// if (auto BinOp = dyn_cast<BinaryOpInterface>(Op)) ...
void BM_TypeTest_Interface(benchmark::State &State, Pattern P) {
  Nodes N(P);
  runBenchmark(State, P, N.Ptrs, [](Operation *Op) -> std::int64_t {
    if (auto BinOp = dyn_cast<BinaryOpInterface>(Op))
      return BinOp.getLHS() + BinOp.getRHS();
    return 0;
  });
}

/// if (Op->hasTrait<IsBinaryOp>()) ...
void BM_TypeTest_HasTrait(benchmark::State &State, Pattern P) {
  Nodes N(P);
  runBenchmark(State, P, N.Ptrs, [](Operation *Op) -> std::int64_t {
    if (Op->hasTrait<IsBinaryOp>())
      return Op->getA() + Op->getB();
    return 0;
  });
}

/// if (isa<AddOp, SubOp, MulOp, DivOp>(Op)) ...
void BM_TypeTest_IsaAnyOf(benchmark::State &State, Pattern P) {
  Nodes N(P);
  runBenchmark(State, P, N.Ptrs, [](Operation *Op) -> std::int64_t {
    if (isa<AddOp, SubOp, MulOp, DivOp>(Op))
      return Op->getA() + Op->getB();
    return 0;
  });
}

} // namespace POLY_NS(mlirstyle)

void POLY_REGISTER_FN(MLIR)(Registry &R) {
  using namespace POLY_NS(mlirstyle);
  R.push_back({Measure::Dispatch, "MLIR_Interface", POLY_INLINE,
               BM_Dispatch_Interface});
  R.push_back({Measure::Dispatch, "MLIR_InterfaceCached", POLY_INLINE,
               BM_Dispatch_InterfaceCached});
  R.push_back({Measure::Dispatch, "MLIR_TypeSwitch", POLY_INLINE,
               BM_Dispatch_TypeSwitch});
  R.push_back({Measure::TypeTest, "MLIR_Interface", POLY_INLINE,
               BM_TypeTest_Interface});
  R.push_back(
      {Measure::TypeTest, "MLIR_HasTrait", POLY_INLINE, BM_TypeTest_HasTrait});
  R.push_back(
      {Measure::TypeTest, "MLIR_IsaAnyOf", POLY_INLINE, BM_TypeTest_IsaAnyOf});
}

} // namespace polybench
