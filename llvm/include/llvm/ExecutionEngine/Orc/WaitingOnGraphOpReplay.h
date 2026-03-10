//===------ WaitingOnGraphOpReplay.h - Record/replay APIs -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utilities for capturing and replaying WaitingOnGraph operations.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_EXECUTIONENGINE_ORC_WAITINGONGRAPHOPREPLAY_H
#define LLVM_EXECUTIONENGINE_ORC_WAITINGONGRAPHOPREPLAY_H

#include "llvm/ADT/fallible_iterator.h"
#include "llvm/ExecutionEngine/Orc/WaitingOnGraph.h"
#include "llvm/Support/Error.h"

#include <mutex>
#include <optional>
#include <variant>

namespace llvm::orc::detail {

/// Records WaitingOnGraph operations to a line-oriented text format on a
/// raw_ostream. The format is a sequence of operations terminated by "end":
///
///   simplify-and-emit <num-supernodes>
///     sn <index>
///       defs <num-containers>
///         container <id> <num-elements>
///           elements <elem-id>...
///       deps <num-containers>
///         ...
///   fail
///     failed <num-containers>
///       container <id> <num-elements>
///         elements <elem-id>...
///   end
///
/// Container and element ids are integers assigned sequentially by the
/// recorder. Leading/trailing whitespace on each line is ignored.
template <typename ContainerIdT, typename ElementIdT>
class WaitingOnGraphOpStreamRecorder
    : public detail::WaitingOnGraph<ContainerIdT, ElementIdT>::OpRecorder {
  using WOG = detail::WaitingOnGraph<ContainerIdT, ElementIdT>;
  using SuperNode = typename WOG::SuperNode;
  using ContainerId = typename WOG::ContainerId;
  using ElementId = typename WOG::ElementId;
  using ContainerElementsMap = typename WOG::ContainerElementsMap;
  using ElementSet = typename WOG::ElementSet;

public:
  WaitingOnGraphOpStreamRecorder(raw_ostream &OS) : OS(OS) {}

  void
  recordSimplify(const std::vector<std::unique_ptr<SuperNode>> &SNs) override {
    std::scoped_lock<std::mutex> Lock(M);
    recordSuperNodes("simplify-and-emit", SNs);
  }

  void recordFail(const ContainerElementsMap &Failed) override {
    std::scoped_lock<std::mutex> Lock(M);
    OS << "fail\n";
    recordContainerElementsMap("  ", "failed", Failed);
  }

  void recordEnd() override {
    std::scoped_lock<std::mutex> Lock(M);
    OS << "end\n";
  }

  // Should render the container id as a string.
  virtual void printContainer(const ContainerId &C) {
    auto I =
        ContainerIdMap.insert(std::make_pair(C, ContainerIdMap.size())).first;
    OS << I->second.Id;
  }

  // Should render the elements of C as a space-separated list (with a space
  // before the first element).
  virtual void printElementsIn(const ContainerId &C,
                               const ElementSet &Elements) {
    assert(ContainerIdMap.count(C));
    auto &ElementIdMap = ContainerIdMap[C].ElementIdMap;
    for (auto &E : Elements) {
      auto I =
          ElementIdMap.insert(std::make_pair(E, ElementIdMap.size())).first;
      OS << " " << I->second;
    }
  }

private:
  struct ContainerIdInfo {
    ContainerIdInfo() = default;
    ContainerIdInfo(size_t Id) : Id(Id) {}
    size_t Id = 0;
    DenseMap<ElementId, size_t> ElementIdMap;
  };
  DenseMap<ContainerId, ContainerIdInfo> ContainerIdMap;

  void recordSuperNodes(StringRef OpName,
                        const std::vector<std::unique_ptr<SuperNode>> &SNs) {
    OS << OpName << " " << SNs.size() << "\n";
    for (size_t I = 0; I != SNs.size(); ++I) {
      OS << "  sn " << I << "\n";
      recordContainerElementsMap("    ", "defs", SNs[I]->defs());
      recordContainerElementsMap("    ", "deps", SNs[I]->deps());
    }
  }

  void recordContainerElementsMap(StringRef Indent, StringRef MapName,
                                  const ContainerElementsMap &M) {
    OS << Indent << MapName << " " << M.size() << "\n";
    for (auto &[Container, Elements] : M) {
      OS << Indent << "  container ";
      printContainer(Container);
      OS << " " << Elements.size() << "\n";
      OS << Indent << "    elements ";
      printElementsIn(Container, Elements);
      OS << "\n";
    }
  }

  std::mutex M;
  raw_ostream &OS;
};

template <typename ContainerIdT, typename ElementIdT>
class WaitingOnGraphOpReplay {
public:
  using Graph = WaitingOnGraph<ContainerIdT, ElementIdT>;
  using SuperNode = typename Graph::SuperNode;
  using ContainerId = typename Graph::ContainerId;
  using ElementId = typename Graph::ElementId;
  using ContainerElementsMap = typename Graph::ContainerElementsMap;
  using ExternalState = typename Graph::ExternalState;

  /// A simplify-and-emit operation parsed from the input.
  struct SimplifyAndEmitOp {
    SimplifyAndEmitOp() = default;
    SimplifyAndEmitOp(SimplifyAndEmitOp &&) = default;
    SimplifyAndEmitOp &operator=(SimplifyAndEmitOp &&) = default;
    SimplifyAndEmitOp(std::vector<std::unique_ptr<SuperNode>> SNs)
        : SNs(std::move(SNs)) {}
    std::vector<std::unique_ptr<SuperNode>> SNs;
  };

  /// A fail operation parsed from the input.
  struct FailOp {
    FailOp() = default;
    FailOp(FailOp &&) = default;
    FailOp &operator=(FailOp &&) = default;
    FailOp(ContainerElementsMap Failed) : Failed(std::move(Failed)) {}
    ContainerElementsMap Failed;
  };

  /// A parsed operation -- either a simplify-and-emit or a fail.
  using Op = std::variant<SimplifyAndEmitOp, FailOp>;

  /// Replay ops on a given graph.
  struct Replayer {
    Replayer(Graph &G) : G(G) {}

    void replay(Op O) {
      if (auto *SimplifyAndEmit = std::get_if<SimplifyAndEmitOp>(&O))
        replaySimplifyAndEmit(std::move(SimplifyAndEmit->SNs));
      else if (auto *Fail = std::get_if<FailOp>(&O))
        replayFail(std::move(Fail->Failed));
    }

    void replaySimplifyAndEmit(std::vector<std::unique_ptr<SuperNode>> SNs) {
      auto SR = Graph::simplify(std::move(SNs));
      auto ER = G.emit(std::move(SR),
                       [this](ContainerId C, ElementId E) -> ExternalState {
                         {
                           auto I = Failed.find(C);
                           if (I != Failed.end() && I->second.count(E))
                             return ExternalState::Failed;
                         }
                         {
                           auto I = Ready.find(C);
                           if (I != Ready.end() && I->second.count(E))
                             return ExternalState::Ready;
                         }
                         return ExternalState::None;
                       });
      for (auto &SN : ER.Ready)
        for (auto &[Container, Elems] : SN->defs())
          Ready[Container].insert(Elems.begin(), Elems.end());
      for (auto &SN : ER.Failed)
        for (auto &[Container, Elems] : SN->defs())
          Failed[Container].insert(Elems.begin(), Elems.end());
    }

    void replayFail(ContainerElementsMap NewlyFailed) {
      for (auto &[Container, Elems] : NewlyFailed)
        Failed[Container].insert(Elems.begin(), Elems.end());

      auto FailedSNs = G.fail(NewlyFailed);
      for (auto &SN : FailedSNs)
        for (auto &[Container, Elems] : SN->defs())
          Failed[Container].insert(Elems.begin(), Elems.end());
    }

    Graph &G;
    ContainerElementsMap Ready;
    ContainerElementsMap Failed;
  };

  /// Parser for input buffer.
  class OpParser {
  public:
    using ParseResult = std::pair<std::optional<Op>, StringRef>;
    virtual ~OpParser() = default;
    virtual Expected<ParseResult> parseNext(StringRef Input) = 0;

  protected:
    Expected<ParseResult>
    parsedSimplifyAndEmit(std::vector<std::unique_ptr<SuperNode>> SNs,
                          StringRef Input) {
      return ParseResult(SimplifyAndEmitOp{std::move(SNs)}, Input);
    }

    Expected<ParseResult> parsedFail(ContainerElementsMap NewlyFailed,
                                     StringRef Input) {
      return ParseResult(FailOp{std::move(NewlyFailed)}, Input);
    }

    Expected<ParseResult> parsedEnd(StringRef Input) {
      return ParseResult(std::nullopt, Input);
    }
  };

  /// Fallible iterator for iterating over WaitingOnGraph ops.
  class OpIterator {
  public:
    /// Default constructed fallible iterator. Serves as end value.
    OpIterator() = default;

    /// Construct a fallible iterator reading from the given input buffer using
    /// the given parser.
    OpIterator(std::shared_ptr<OpParser> P, StringRef Input)
        : P(std::move(P)), Input(Input), PrevInput(Input) {}

    OpIterator(const OpIterator &Other)
        : P(Other.P), Input(Other.PrevInput), PrevInput(Other.PrevInput) {
      // We can't just copy Op, we need to re-parse.
      if (this->P)
        cantFail(inc());
    }

    OpIterator &operator=(const OpIterator &Other) {
      P = Other.P;
      Input = PrevInput = Other.PrevInput;
      if (this->P)
        cantFail(inc());
      return *this;
    }

    OpIterator(OpIterator &&) = default;
    OpIterator &operator=(OpIterator &&) = default;

    /// Move to next record.
    Error inc() {
      PrevInput = Input;
      auto PR = P->parseNext(Input);
      if (!PR)
        return PR.takeError();
      std::tie(CurOp, Input) = std::move(*PR);
      if (!CurOp) {
        P = nullptr;
        Input = "";
      }
      return Error::success();
    }

    // Dereference. Note: Moves op type.
    Op &operator*() {
      assert(CurOp && "Dereferencing end/invalid iterator");
      return *CurOp;
    }

    // Dereference. Note: Moves op type.
    const Op &operator*() const {
      assert(CurOp && "Dereferencing end/invalid iterator");
      return *CurOp;
    }

    /// Compare iterators. End iterators compare equal.
    friend bool operator==(const OpIterator &LHS, const OpIterator &RHS) {
      return LHS.P == RHS.P && LHS.Input == RHS.Input;
    }

    friend bool operator!=(const OpIterator &LHS, const OpIterator &RHS) {
      return !(LHS == RHS);
    }

  private:
    std::shared_ptr<OpParser> P;
    StringRef Input, PrevInput;
    std::optional<Op> CurOp;
  };
};

/// Returns a fallible iterator range over the operations in the given buffer.
/// The buffer should contain text in the format produced by
/// WaitingOnGraphOpStreamRecorder. Parsing errors are reported through Err.
template <typename ContainerIdT, typename ElementIdT>
iterator_range<fallible_iterator<
    typename WaitingOnGraphOpReplay<ContainerIdT, ElementIdT>::OpIterator>>
readWaitingOnGraphOpsFromBuffer(StringRef InputBuffer, Error &Err) {

  using Replay = WaitingOnGraphOpReplay<ContainerIdT, ElementIdT>;

  class Parser : public Replay::OpParser {
  public:
    using ParseResult = typename Replay::OpParser::ParseResult;
    using SuperNode = typename Replay::SuperNode;
    using ContainerElementsMap = typename Replay::ContainerElementsMap;

    /// Parse the next operation from Input into CurrentOp.
    /// Sets IsEnd on "end" keyword. Returns Error on parse failure.
    Expected<ParseResult> parseNext(StringRef Input) override {
      auto Line = getNextLine(Input);

      if (Line.empty())
        return make_error<StringError>(
            "unexpected end of input (missing 'end')",
            inconvertibleErrorCode());

      if (Line.consume_front("simplify-and-emit ")) {
        size_t NumSNs;
        if (Line.trim().consumeInteger(10, NumSNs))
          return make_error<StringError>(
              "expected supernode count after 'simplify-and-emit'",
              inconvertibleErrorCode());
        auto SNs = parseSuperNodes(Input, NumSNs);
        if (!SNs)
          return SNs.takeError();
        return this->parsedSimplifyAndEmit(std::move(*SNs), Input);
      } else if (Line.trim() == "fail") {
        auto FailElems = parseContainerElementsMap("failed", Input);
        if (!FailElems)
          return FailElems.takeError();
        return this->parsedFail(std::move(*FailElems), Input);
      } else if (Line.trim() == "end")
        return this->parsedEnd(Input);
      else
        return make_error<StringError>("unexpected line: '" + Line + "'",
                                       inconvertibleErrorCode());
    }

  private:
    static StringRef getNextLine(StringRef &Input) {
      StringRef Line;
      // Parse skipping blank lines.
      do {
        std::tie(Line, Input) = Input.split('\n');
        Line = Line.trim();
      } while (Line.empty() && !Input.empty());
      return Line;
    }

    static Expected<std::vector<std::unique_ptr<SuperNode>>>
    parseSuperNodes(StringRef &Input, size_t NumSNs) {
      std::vector<std::unique_ptr<SuperNode>> SNs;
      for (size_t I = 0; I != NumSNs; ++I) {
        // Parse "sn <index>"
        StringRef Line = getNextLine(Input);
        if (!Line.consume_front("sn "))
          return make_error<StringError>("expected 'sn " + Twine(I) + "'",
                                         inconvertibleErrorCode());

        auto Defs = parseContainerElementsMap("defs", Input);
        if (!Defs)
          return Defs.takeError();
        auto Deps = parseContainerElementsMap("deps", Input);
        if (!Deps)
          return Deps.takeError();

        SNs.push_back(
            std::make_unique<SuperNode>(std::move(*Defs), std::move(*Deps)));
      }
      return std::move(SNs);
    }

    static Expected<ContainerElementsMap>
    parseContainerElementsMap(StringRef ArgName, StringRef &Input) {
      // Parse "defs <count>"
      auto Line = getNextLine(Input);
      if (!Line.consume_front(ArgName))
        return make_error<StringError>("expected '" + ArgName + " <count>'",
                                       inconvertibleErrorCode());
      size_t NumContainers;
      if (Line.trim().consumeInteger(10, NumContainers))
        return make_error<StringError>("expected " + ArgName + " count",
                                       inconvertibleErrorCode());

      ContainerElementsMap M;
      for (size_t I = 0; I != NumContainers; ++I) {
        Line = getNextLine(Input);
        if (!Line.consume_front("container "))
          return make_error<StringError>("expected 'container <id> <count>'",
                                         inconvertibleErrorCode());

        size_t Container;
        Line = Line.trim();
        if (Line.consumeInteger(10, Container))
          return make_error<StringError>("expected container id",
                                         inconvertibleErrorCode());

        if (M.count(Container))
          return make_error<StringError>(
              "expected container id to be unique within " + ArgName,
              inconvertibleErrorCode());

        size_t NumElements;
        if (Line.trim().consumeInteger(10, NumElements))
          return make_error<StringError>("expected elements count",
                                         inconvertibleErrorCode());
        if (NumElements == 0)
          return make_error<StringError>("number of elements for container " +
                                             Twine(Container) + " must be > 0",
                                         inconvertibleErrorCode());

        Line = getNextLine(Input);
        if (!Line.consume_front("elements "))
          return make_error<StringError>("expected 'elements ...'",
                                         inconvertibleErrorCode());

        auto &Elements = M[Container];
        for (size_t J = 0; J != NumElements; ++J) {
          size_t Elem;
          Line = Line.trim();
          if (Line.consumeInteger(10, Elem))
            return make_error<StringError>("expected element id",
                                           inconvertibleErrorCode());
          if (Elements.count(Elem))
            return make_error<StringError>(
                "expected element id to be unique within container " +
                    Twine(Container),
                inconvertibleErrorCode());
          Elements.insert(Elem);
        }
      }

      return std::move(M);
    }
  };

  ErrorAsOutParameter _(Err);
  typename Replay::OpIterator Begin(std::make_shared<Parser>(), InputBuffer);
  typename Replay::OpIterator End;
  if ((Err = Begin.inc())) // Parse first operation.
    Begin = End;
  return make_fallible_range(std::move(Begin), std::move(End), Err);
}

} // namespace llvm::orc::detail

#endif // LLVM_EXECUTIONENGINE_ORC_WAITINGONGRAPHOPREPLAY_H
