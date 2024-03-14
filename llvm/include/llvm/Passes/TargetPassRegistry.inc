//===- TargetPassRegistry.inc - Registry of passes --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is used as the registry of passes in registerPassBuilderCallbacks
//
//===----------------------------------------------------------------------===//

// NOTE: NO INCLUDE GUARD DESIRED!

#ifdef GET_PASS_REGISTRY

#if !__has_include(GET_PASS_REGISTRY)
#error "must provide <Target>PassRegistry.def"
#endif

{

  // FIXME: use std::size when MSVC_MIN >= 19.30

  static constexpr bool HAVE_MODULE_ANALYSIS_ =
      std::array{
          "",
#define MODULE_ANALYSIS(NAME, CREATE_PASS) NAME,
#include GET_PASS_REGISTRY
#undef MODULE_ANALYSIS
      }
          .size() > 1;

  static constexpr bool HAVE_MODULE_PASS_ =
      std::array{
          "",
#define MODULE_PASS(NAME, CREATE_PASS) NAME,
#include GET_PASS_REGISTRY
#undef MODULE_PASS
      }
          .size() > 1;

  static constexpr bool HAVE_MODULE_PASS_WITH_PARAMS_ =
      std::array{
          "",
#define MODULE_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS) NAME,
#include GET_PASS_REGISTRY
#undef MODULE_PASS_WITH_PARAMS
      }
          .size() > 1;

  static constexpr bool HAVE_FUNCTION_ANALYSIS_ =
      std::array{
          "",
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS) NAME,
#include GET_PASS_REGISTRY
#undef FUNCTION_ANALYSIS
      }
          .size() > 1;

  static constexpr bool HAVE_FUNCTION_ALIAS_ANALYSIS_ =
      std::array{
          "",
#define FUNCTION_ALIAS_ANALYSIS(NAME, CREATE_PASS) NAME,
#include GET_PASS_REGISTRY
#undef FUNCTION_ALIAS_ANALYSIS
      }
          .size() > 1;

  static constexpr bool HAVE_FUNCTION_PASS_ =
      std::array{
          "",
#define FUNCTION_PASS(NAME, CREATE_PASS) NAME,
#include GET_PASS_REGISTRY
#undef FUNCTION_PASS
      }
          .size() > 1;

  static constexpr bool HAVE_FUNCTION_PASS_WITH_PARAMS_ =
      std::array{
          "",
#define FUNCTION_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)    \
  NAME,
#include GET_PASS_REGISTRY
#undef FUNCTION_PASS_WITH_PARAMS
      }
          .size() > 1;

  static constexpr bool HAVE_LOOP_ANALYSIS_ =
      std::array{
          "",
#define LOOP_ANALYSIS(NAME, CREATE_PASS) NAME,
#include GET_PASS_REGISTRY
#undef LOOP_ANALYSIS
      }
          .size() > 1;

  static constexpr bool HAVE_LOOP_PASS_ =
      std::array{
          "",
#define LOOP_PASS(NAME, CREATE_PASS) NAME,
#include GET_PASS_REGISTRY
#undef LOOP_PASS
      }
          .size() > 1;

  static constexpr bool HAVE_MACHINE_FUNCTION_ANALYSIS_ =
      std::array{
          "",
#define MACHINE_FUNCTION_ANALYSIS(NAME, CREATE_PASS) NAME,
#include GET_PASS_REGISTRY
#undef MACHINE_FUNCTION_ANALYSIS
      }
          .size() > 1;

  static constexpr bool HAVE_MACHINE_FUNCTION_PASS_ =
      std::array{
          "",
#define MACHINE_FUNCTION_PASS(NAME, CREATE_PASS) NAME,
#include GET_PASS_REGISTRY
#undef MACHINE_FUNCTION_PASS
      }
          .size() > 1;

  static constexpr bool HAVE_MACHINE_FUNCTION_PASS_WITH_PARAMS_ =
      std::array{
          "",
#define MACHINE_FUNCTION_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER,    \
                                          PARAMS)                              \
  NAME,
#include GET_PASS_REGISTRY
#undef MACHINE_FUNCTION_PASS_WITH_PARAMS
      }
          .size() > 1;

  if (PopulateClassToPassNames) {
    auto *PIC = PB.getPassInstrumentationCallbacks();

#define ADD_CLASS_PASS_TO_PASS_NAME(NAME, CREATE_PASS)                         \
  PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#define ADD_CLASS_PASS_TO_PASS_NAME_WITH_PARAMS(NAME, CLASS)                   \
  PIC->addClassToPassName(CLASS, NAME);

#define MODULE_ANALYSIS(NAME, CREATE_PASS)                                     \
  ADD_CLASS_PASS_TO_PASS_NAME(NAME, CREATE_PASS)
#define MODULE_PASS(NAME, CREATE_PASS)                                         \
  ADD_CLASS_PASS_TO_PASS_NAME(NAME, CREATE_PASS)
#define MODULE_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)      \
  ADD_CLASS_PASS_TO_PASS_NAME_WITH_PARAMS(NAME, CLASS)
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  ADD_CLASS_PASS_TO_PASS_NAME(NAME, CREATE_PASS)
#define FUNCTION_ALIAS_ANALYSIS(NAME, CREATE_PASS)                             \
  ADD_CLASS_PASS_TO_PASS_NAME(NAME, CREATE_PASS)
#define FUNCTION_PASS(NAME, CREATE_PASS)                                       \
  ADD_CLASS_PASS_TO_PASS_NAME(NAME, CREATE_PASS)
#define FUNCTION_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)    \
  ADD_CLASS_PASS_TO_PASS_NAME_WITH_PARAMS(NAME, CLASS)
#define LOOP_ANALYSIS(NAME, CREATE_PASS)                                       \
  ADD_CLASS_PASS_TO_PASS_NAME(NAME, CREATE_PASS)
#define LOOP_PASS(NAME, CREATE_PASS)                                           \
  ADD_CLASS_PASS_TO_PASS_NAME(NAME, CREATE_PASS)
#define MACHINE_FUNCTION_ANALYSIS(NAME, CREATE_PASS)                           \
  ADD_CLASS_PASS_TO_PASS_NAME(NAME, CREATE_PASS)
#define MACHINE_FUNCTION_PASS(NAME, CREATE_PASS)                               \
  ADD_CLASS_PASS_TO_PASS_NAME(NAME, CREATE_PASS)
#define MACHINE_FUNCTION_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER,    \
                                          PARAMS)                              \
  ADD_CLASS_PASS_TO_PASS_NAME_WITH_PARAMS(NAME, CREATE_PASS)
#include GET_PASS_REGISTRY
#undef MODULE_ANALYSIS
#undef MODULE_PASS
#undef MODULE_PASS_WITH_PARAMS
#undef FUNCTION_ANALYSIS
#undef FUNCTION_PASS
#undef FUNCTION_PASS_WITH_PARAMS
#undef LOOP_ANALYSIS
#undef LOOP_PASS
#undef MACHINE_FUNCTION_ANALYSIS
#undef MACHINE_FUNCTION_PASS
#undef MACHINE_FUNCTION_PASS_WITH_PARAMS
#undef ADD_CLASS_PASS_TO_PASS_NAME
#undef ADD_CLASS_PASS_TO_PASS_NAME_WITH_PARAMS
  }

#define ADD_PASS(NAME, CREATE_PASS)                                            \
  if (Name == NAME) {                                                          \
    PM.addPass(CREATE_PASS);                                                   \
    return true;                                                               \
  }

#define ADD_PASS_WITH_PARAMS(NAME, CREATE_PASS, PARSER)                        \
  if (PassBuilder::checkParametrizedPassName(Name, NAME)) {                    \
    auto Params = PassBuilder::parsePassParameters(PARSER, Name, NAME);        \
    if (!Params)                                                               \
      return false;                                                            \
    PM.addPass(CREATE_PASS(Params.get()));                                     \
    return true;                                                               \
  }

  if constexpr (HAVE_MODULE_PASS_) {
    PB.registerPipelineParsingCallback(
        [=](StringRef Name, ModulePassManager &PM,
            ArrayRef<PassBuilder::PipelineElement>) {
#define MODULE_PASS(NAME, CREATE_PASS) ADD_PASS(NAME, CREATE_PASS)
#include GET_PASS_REGISTRY
#undef MODULE_PASS
          return false;
        });
  }

  if constexpr (HAVE_MODULE_PASS_WITH_PARAMS_) {
    PB.registerPipelineParsingCallback(
        [=](StringRef Name, ModulePassManager &PM,
            ArrayRef<PassBuilder::PipelineElement>) {
#define MODULE_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)      \
  ADD_PASS_WITH_PARAMS(NAME, CREATE_PASS, PARSER)
#include GET_PASS_REGISTRY
#undef MODULE_PASS_WITH_PARAMS
          return false;
        });
  }

  if constexpr (HAVE_FUNCTION_PASS_) {
    PB.registerPipelineParsingCallback(
        [=](StringRef Name, FunctionPassManager &PM,
            ArrayRef<PassBuilder::PipelineElement>) {
#define FUNCTION_PASS(NAME, CREATE_PASS) ADD_PASS(NAME, CREATE_PASS)
#include GET_PASS_REGISTRY
#undef FUNCTION_PASS
          return false;
        });
  }

  if constexpr (HAVE_FUNCTION_PASS_WITH_PARAMS_) {
    PB.registerPipelineParsingCallback(
        [=](StringRef Name, FunctionPassManager &PM,
            ArrayRef<PassBuilder::PipelineElement>) {
#define FUNCTION_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER, PARAMS)    \
  ADD_PASS_WITH_PARAMS(NAME, CREATE_PASS, PARSER)
#include GET_PASS_REGISTRY
#undef FUNCTION_PASS_WITH_PARAMS
          return false;
        });
  }

  if constexpr (HAVE_LOOP_PASS_) {
    PB.registerPipelineParsingCallback(
        [=](StringRef Name, LoopPassManager &PM,
            ArrayRef<PassBuilder::PipelineElement>) {
#define LOOP_PASS(NAME, CREATE_PASS) ADD_PASS(NAME, CREATE_PASS)
#include GET_PASS_REGISTRY
          return false;
        });
  }

  if constexpr (HAVE_MACHINE_FUNCTION_PASS_) {
    PB.registerPipelineParsingCallback(
        [=](StringRef Name, MachineFunctionPassManager &PM,
            ArrayRef<PassBuilder::PipelineElement>) {
#define MACHINE_FUNCTION_PASS(NAME, CREATE_PASS) ADD_PASS(NAME, CREATE_PASS)
#include GET_PASS_REGISTRY
          return false;
        });
  }

  if constexpr (HAVE_MACHINE_FUNCTION_PASS_WITH_PARAMS_) {
    PB.registerPipelineParsingCallback(
        [=](StringRef Name, FunctionPassManager &PM,
            ArrayRef<PassBuilder::PipelineElement>) {
#define MACHINE_FUNCTION_PASS_WITH_PARAMS(NAME, CLASS, CREATE_PASS, PARSER,    \
                                          PARAMS)                              \
  ADD_PASS_WITH_PARAMS(NAME, CREATE_PASS, PARSER)
#include GET_PASS_REGISTRY
#undef MACHINE_FUNCTION_PASS_WITH_PARAMS
          return false;
        });
  }
#undef ADD_PASS
#undef ADD_PASS_WITH_PARAMS

  if constexpr (HAVE_MODULE_ANALYSIS_) {
    PB.registerAnalysisRegistrationCallback([](ModuleAnalysisManager &AM) {
#define MODULE_ANALYSIS(NAME, CREATE_PASS)                                     \
  AM.registerPass([&] { return CREATE_PASS; });
#include GET_PASS_REGISTRY
#undef MODULE_ANALYSIS
    });
  }

  if constexpr (HAVE_FUNCTION_ANALYSIS_) {
    PB.registerAnalysisRegistrationCallback([](FunctionAnalysisManager &AM) {
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  AM.registerPass([&] { return CREATE_PASS; });
#include GET_PASS_REGISTRY
#undef FUNCTION_ANALYSIS
    });
  }

  if constexpr (HAVE_FUNCTION_ALIAS_ANALYSIS_) {
    PB.registerParseAACallback([](StringRef Name, AAManager &AM) {
#define FUNCTION_ANALYSIS(NAME, CREATE_PASS)                                   \
  if (Name == NAME) {                                                          \
    AM.registerFunctionAnalysis<                                               \
        std::remove_reference_t<decltype(CREATE_PASS)>>();                     \
    return true;                                                               \
  }
#include GET_PASS_REGISTRY
#undef FUNCTION_ANALYSIS
      return false;
    });
  }

  if constexpr (HAVE_LOOP_ANALYSIS_) {
    PB.registerAnalysisRegistrationCallback([](LoopAnalysisManager &AM) {
#define LOOP_ANALYSIS(NAME, CREATE_PASS)                                       \
  AM.registerPass([&] { return CREATE_PASS; });
#include GET_PASS_REGISTRY
#undef LOOP_ANALYSIS
    });
  }

  if constexpr (HAVE_MACHINE_FUNCTION_ANALYSIS_) {
    PB.registerAnalysisRegistrationCallback(
        [](MachineFunctionAnalysisManager &AM) {
#define MACHINE_FUNCTION_ANALYSIS(NAME, CREATE_PASS)                           \
  AM.registerPass([&] { return CREATE_PASS; });
#include GET_PASS_REGISTRY
#undef MACHINE_FUNCTION_ANALYSIS
        });
  }
}

#undef GET_PASS_REGISTRY
#endif // GET_PASS_REGISTRY
