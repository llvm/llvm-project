# Unity build exclusions.
# This file centralizes all SKIP_UNITY_BUILD_INCLUSION properties
# that were previously scattered across individual CMakeLists.txt files.
#
# Include this file from the top-level CMakeLists.txt after all targets
# have been defined.

# ======================================================================
# clang-tools-extra
# ======================================================================

if(TARGET clangDoc)
  # From clang-tools-extra/clang-doc/CMakeLists.txt (target: clangDoc)
  # Representation.cpp and YAMLGenerator.cpp have conflicting static declarations
  # and CommentKind ambiguity when combined in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang-tools-extra/clang-doc/Representation.cpp
    ${LLVM_REPO_DIR}/clang-tools-extra/clang-doc/YAMLGenerator.cpp
    TARGET_DIRECTORY clangDoc
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET clangDocSupport)
  # From clang-tools-extra/clang-doc/support/CMakeLists.txt (target: clangDocSupport)
  # Utils.cpp references ClangDocContext which is not visible when combined
  # with File.cpp in a unity build due to include ordering.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang-tools-extra/clang-doc/support/Utils.cpp
    TARGET_DIRECTORY clangDocSupport
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET clangDaemon)
  # From clang-tools-extra/clangd/CMakeLists.txt (target: clangDaemon)
  # clangd::Token conflicts with clang::Token, and InsertionPoint.cpp has
  # type conflicts when combined with other files in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang-tools-extra/clangd/SourceCode.cpp
    ${LLVM_REPO_DIR}/clang-tools-extra/clangd/refactor/InsertionPoint.cpp
    TARGET_DIRECTORY clangDaemon
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET clangDaemonTweaks)
  # From clang-tools-extra/clangd/refactor/tweaks/CMakeLists.txt (target: clangDaemonTweaks)
  # DefineInline/DefineOutline share static getSelectedFunction;
  # RemoveUsingNamespace has clangd::Token vs clang::Token conflicts.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang-tools-extra/clangd/refactor/tweaks/DefineInline.cpp
    ${LLVM_REPO_DIR}/clang-tools-extra/clangd/refactor/tweaks/DefineOutline.cpp
    ${LLVM_REPO_DIR}/clang-tools-extra/clangd/refactor/tweaks/RemoveUsingNamespace.cpp
    TARGET_DIRECTORY clangDaemonTweaks
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET ClangdTests)
  # From clang-tools-extra/clangd/unittests/CMakeLists.txt (target: ClangdTests)
  # These files define symbols that clash in unity builds:
  # - PreambleTests.cpp defines MATCHER_P(Diag, ...) which conflicts with
  #   clang::clangd::Diag when combined with RenameTests.cpp.
  # - HoverTests.cpp and IncludeCleanerTests.cpp both define guard().
  # - SymbolCollectorTests.cpp, SerializationTests.cpp, FileIndexTests.cpp,
  #   BackgroundIndexTests.cpp, and FindSymbolsTests.cpp all define
  #   MATCHER_P(qName, ...).
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang-tools-extra/clangd/unittests/PreambleTests.cpp
    ${LLVM_REPO_DIR}/clang-tools-extra/clangd/unittests/HoverTests.cpp
    ${LLVM_REPO_DIR}/clang-tools-extra/clangd/unittests/IncludeCleanerTests.cpp
    ${LLVM_REPO_DIR}/clang-tools-extra/clangd/unittests/SymbolCollectorTests.cpp
    TARGET_DIRECTORY ClangdTests
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET ClangIncludeCleanerTests)
  # From clang-tools-extra/include-cleaner/unittests/CMakeLists.txt (target: ClangIncludeCleanerTests)
  # AnalysisTest.cpp and FindHeadersTest.cpp both define guard().
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang-tools-extra/include-cleaner/unittests/FindHeadersTest.cpp
    TARGET_DIRECTORY ClangIncludeCleanerTests
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET modularize)
  # From clang-tools-extra/modularize/CMakeLists.txt (target: modularize)
  # ModuleAssistant.cpp defines a local Module class that conflicts with
  # clang::Module when combined in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang-tools-extra/modularize/ModuleAssistant.cpp
    TARGET_DIRECTORY modularize
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET ClangTidyTests)
  # From clang-tools-extra/unittests/clang-tidy/CMakeLists.txt (target: clang-tidy)
  # These files define symbols that clash in unity builds:
  # - ClangTidyOptionsTest.cpp and ClangTidyDiagnosticConsumerTest.cpp both
  #   define TestCheck
  # - UsingInserterTest.cpp and NamespaceAliaserTest.cpp both define runChecker
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang-tools-extra/unittests/clang-tidy/ClangTidyDiagnosticConsumerTest.cpp
    ${LLVM_REPO_DIR}/clang-tools-extra/unittests/clang-tidy/ClangTidyOptionsTest.cpp
    ${LLVM_REPO_DIR}/clang-tools-extra/unittests/clang-tidy/NamespaceAliaserTest.cpp
    ${LLVM_REPO_DIR}/clang-tools-extra/unittests/clang-tidy/UsingInserterTest.cpp
    TARGET_DIRECTORY ClangTidyTests
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

# ======================================================================
# clang
# ======================================================================

if(TARGET clangAPINotes)
  # From clang/lib/APINotes/CMakeLists.txt (target: clangAPINotes)
  # Reader and Writer define identically-named classes in anonymous namespaces
  # (IdentifierTableInfo, ContextIDTableInfo, etc.). Manager and YAMLCompiler
  # use "using namespace api_notes" which brings api_notes::Module into scope,
  # conflicting with clang::Module.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang/lib/APINotes/APINotesReader.cpp
    ${LLVM_REPO_DIR}/clang/lib/APINotes/APINotesWriter.cpp
    ${LLVM_REPO_DIR}/clang/lib/APINotes/APINotesYAMLCompiler.cpp
    TARGET_DIRECTORY clangAPINotes
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET clangAST)
  # From clang/lib/AST/CMakeLists.txt (target: clangAST)
  # CommentParser/CommentSema use "using namespace clang::comments" which
  # conflicts with clang::tok. MicrosoftMangle and ItaniumMangle define
  # functions with the same name that cause overload ambiguity.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang/lib/AST/CommentParser.cpp
    ${LLVM_REPO_DIR}/clang/lib/AST/CommentSema.cpp
    ${LLVM_REPO_DIR}/clang/lib/AST/MicrosoftMangle.cpp
    ${LLVM_REPO_DIR}/clang/lib/AST/ItaniumMangle.cpp
    TARGET_DIRECTORY clangAST
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET clangAnalysisLifetimeSafety)
  # From clang/lib/Analysis/LifetimeSafety/CMakeLists.txt (target: clangAnalysisLifetimeSafety)
  # Multiple files define identically-named struct Lattice and class AnalysisImpl
  # in anonymous namespaces, causing redefinition errors in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang/lib/Analysis/LifetimeSafety/LoanPropagation.cpp
    ${LLVM_REPO_DIR}/clang/lib/Analysis/LifetimeSafety/MovedLoans.cpp
    TARGET_DIRECTORY clangAnalysisLifetimeSafety
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET clangBasic)
  # From clang/lib/Basic/CMakeLists.txt (target: clangBasic)
  # Each Targets/*.cpp file defines static constexpr NumBuiltins, BuiltinInfos,
  # and BuiltinStrings which conflict in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang/lib/Basic/Targets/AMDGPU.cpp
    ${LLVM_REPO_DIR}/clang/lib/Basic/Targets/ARM.cpp
    ${LLVM_REPO_DIR}/clang/lib/Basic/Targets/BPF.cpp
    ${LLVM_REPO_DIR}/clang/lib/Basic/Targets/DirectX.cpp
    ${LLVM_REPO_DIR}/clang/lib/Basic/Targets/Hexagon.cpp
    ${LLVM_REPO_DIR}/clang/lib/Basic/Targets/LoongArch.cpp
    ${LLVM_REPO_DIR}/clang/lib/Basic/Targets/Mips.cpp
    ${LLVM_REPO_DIR}/clang/lib/Basic/Targets/NVPTX.cpp
    ${LLVM_REPO_DIR}/clang/lib/Basic/Targets/PPC.cpp
    ${LLVM_REPO_DIR}/clang/lib/Basic/Targets/RISCV.cpp
    ${LLVM_REPO_DIR}/clang/lib/Basic/Targets/SPIR.cpp
    ${LLVM_REPO_DIR}/clang/lib/Basic/Targets/SystemZ.cpp
    ${LLVM_REPO_DIR}/clang/lib/Basic/Targets/VE.cpp
    ${LLVM_REPO_DIR}/clang/lib/Basic/Targets/WebAssembly.cpp
    ${LLVM_REPO_DIR}/clang/lib/Basic/Targets/X86.cpp
    ${LLVM_REPO_DIR}/clang/lib/Basic/Targets/XCore.cpp
    TARGET_DIRECTORY clangBasic
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET clangCIR)
  set_source_files_properties(
    # From clang/lib/CIR/CodeGen/CMakeLists.txt (target: clangCIR)
    # CIRGenExprComplex.cpp references both cir::ComplexType/CastKind and
    # clang::ComplexType/CastKind which become ambiguous in unity builds.
    ${LLVM_REPO_DIR}/clang/lib/CIR/CodeGen/CIRGenExprComplex.cpp
    # From clang/lib/CIR/CodeGen/CMakeLists.txt (target: clangCIR)
    # CIRGenOpenACCRecipe.h has no include guards and defines classes; including it
    # from both CIRGenOpenACCClause.cpp and CIRGenOpenACCRecipe.cpp in the same
    # unity TU causes redefinition errors.
    ${LLVM_REPO_DIR}/clang/lib/CIR/CodeGen/CIRGenOpenACCClause.cpp
    ${LLVM_REPO_DIR}/clang/lib/CIR/CodeGen/CIRGenOpenACCRecipe.cpp
    # From clang/lib/CIR/CodeGen/CMakeLists.txt (target: clangCIR)
    # CIRGenTypes.cpp references cir::Type/RecordType which conflict with
    # clang::Type/RecordType from other files in the unity TU.
    ${LLVM_REPO_DIR}/clang/lib/CIR/CodeGen/CIRGenTypes.cpp
    TARGET_DIRECTORY clangCIR
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET clangCodeGen)
  # From clang/lib/CodeGen/CMakeLists.txt (target: clangCodeGen)
  # Many CodeGen files use "using namespace llvm" which brings llvm::Type,
  # llvm::VectorType, etc. into scope, conflicting with clang:: equivalents
  # used by other files in the same unity TU.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/BackendUtil.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/CGAtomic.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/CGBuiltin.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/CGCoroutine.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/CGExpr.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/CGHLSLBuiltins.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/CGHLSLRuntime.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/CGLoopInfo.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/CGNonTrivialStruct.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/CGOpenMPRuntime.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/CGOpenMPRuntimeGPU.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/CGStmtOpenMP.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/CodeGenAction.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/CodeGenFunction.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/CodeGenModule.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/CodeGenPGO.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/CoverageMappingGen.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/LinkInModulesPass.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/QualTypeMapper.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/TargetBuiltins/AMDGPU.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/TargetBuiltins/ARM.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/TargetBuiltins/DirectX.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/TargetBuiltins/Hexagon.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/TargetBuiltins/NVPTX.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/TargetBuiltins/PPC.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/TargetBuiltins/RISCV.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/TargetBuiltins/SPIR.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/TargetBuiltins/SystemZ.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/TargetBuiltins/WebAssembly.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/TargetBuiltins/X86.cpp
    ${LLVM_REPO_DIR}/clang/lib/CodeGen/Targets/AArch64.cpp
    TARGET_DIRECTORY clangCodeGen
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET clangExtractAPI)
  # From clang/lib/ExtractAPI/CMakeLists.txt (target: clangExtractAPI)
  # SymbolGraphSerializer.cpp uses "using namespace llvm" which brings types
  # that conflict with clang::extractapi types in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang/lib/ExtractAPI/Serialization/SymbolGraphSerializer.cpp
    TARGET_DIRECTORY clangExtractAPI
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET clangFrontend)
  # From clang/lib/Frontend/CMakeLists.txt (target: clangFrontend)
  # config.h uses a #include guard that errors on double-inclusion, which
  # breaks unity builds when multiple .cpp files include it.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang/lib/Frontend/CompilerInstance.cpp
    ${LLVM_REPO_DIR}/clang/lib/Frontend/CompilerInvocation.cpp
    TARGET_DIRECTORY clangFrontend
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET clangRewriteFrontend)
  # From clang/lib/Frontend/Rewrite/CMakeLists.txt (target: clangRewriteFrontend)
  # Both files include config.h which uses #error to prevent double-inclusion,
  # breaking unity builds when they end up in the same translation unit.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang/lib/Frontend/Rewrite/RewriteModernObjC.cpp
    ${LLVM_REPO_DIR}/clang/lib/Frontend/Rewrite/RewriteObjC.cpp
    TARGET_DIRECTORY clangRewriteFrontend
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET clangLex)
  # From clang/lib/Lex/CMakeLists.txt (target: clangLex)
  # "using namespace clang::dependency_directives_scan" at file scope brings
  # dependency_directives_scan::Token into scope, conflicting with clang::Token.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang/lib/Lex/DependencyDirectivesScanner.cpp
    TARGET_DIRECTORY clangLex
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET clangScalableStaticAnalysisFrameworkCore)
  # From clang/lib/ScalableStaticAnalysisFramework/Core/CMakeLists.txt (target: clangScalableStaticAnalysisFrameworkCore)
  # error: type alias redefinition with different types ('Registry<AnalysisBase>' vs 'Registry<SummaryDataBuilderBase>')
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang/lib/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisRegistry.cpp
    TARGET_DIRECTORY clangScalableStaticAnalysisFrameworkCore
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET clangSema)
  # From clang/lib/Sema/CMakeLists.txt (target: clangSema)
  # Both files define struct PartialSpecMatchResult in anonymous namespaces
  # with different member types, causing redefinition errors in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang/lib/Sema/SemaTemplateInstantiate.cpp
    TARGET_DIRECTORY clangSema
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET clangSerialization)
  # From clang/lib/Serialization/CMakeLists.txt (target: clangSerialization)
  # ASTWriter.cpp defines HeaderFileInfoTrait and LazySpecializationInfoLookupTrait
  # in an anonymous namespace, conflicting with identically-named classes from
  # ASTReaderInternals.h included by other files in this directory.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang/lib/Serialization/ASTWriter.cpp
    TARGET_DIRECTORY clangSerialization
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET clangTooling)
  # From clang/lib/Tooling/CMakeLists.txt (target: clangTooling)
  # CompilationDatabase.cpp and DependencyScanningTool.cpp reference
  # clang::driver which becomes ambiguous when combined with other files
  # in a unity build.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang/lib/Tooling/CompilationDatabase.cpp
    ${LLVM_REPO_DIR}/clang/lib/Tooling/DependencyScanningTool.cpp
    ${LLVM_REPO_DIR}/clang/lib/Tooling/Tooling.cpp
    TARGET_DIRECTORY clangTooling
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET clangTransformer)
  # From clang/lib/Tooling/Transformer/CMakeLists.txt (target: clangTransformer)
  # Both files define a static getNode() helper in anonymous namespaces,
  # causing redefinition errors in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang/lib/Tooling/Transformer/Stencil.cpp
    ${LLVM_REPO_DIR}/clang/lib/Tooling/Transformer/RangeSelector.cpp
    TARGET_DIRECTORY clangTransformer
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET clang-installapi)
  # From clang/tools/clang-installapi/CMakeLists.txt (target: clang-installapi)
  # Options.cpp uses "using namespace llvm" which makes llvm::Value ambiguous
  # with other types when combined in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang/tools/clang-installapi/Options.cpp
    TARGET_DIRECTORY clang-installapi
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET diagtool)
  # From clang/tools/diagtool/CMakeLists.txt (target: diagtool)
  # TreeView.cpp defines printUsage which conflicts with identically-named
  # functions from other diagtool files in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang/tools/diagtool/TreeView.cpp
    TARGET_DIRECTORY diagtool
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET clang)
  # From clang/tools/driver/CMakeLists.txt (target: clang)
  # cc1as_main.cpp and driver.cpp both define LLVMErrorHandler; cc1gen_reproducer
  # uses "driver" which is ambiguous with clang::driver in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang/tools/driver/cc1_main.cpp
    ${LLVM_REPO_DIR}/clang/tools/driver/cc1as_main.cpp
    ${LLVM_REPO_DIR}/clang/tools/driver/cc1gen_reproducer_main.cpp
    TARGET_DIRECTORY clang
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET AllClangUnitTests)
  # From clang/unittests/CMakeLists.txt (target: AllClangUnitTests)
  # Files that clash in unity builds of AllClangUnitTests due to shared headers
  # (ASTPrint.h, CheckerRegistration.h, CallbacksCommon.h, CFGBuildResult.h)
  # or duplicate symbols.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang/unittests/Lex/LexerTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Analysis/CFGBackEdgesTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Analysis/CFGDominatorTree.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Analysis/CFGTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Analysis/IntervalPartitionTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Analysis/UnsafeBufferUsageTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Analysis/FlowSensitive/SingleVarConstantPropagationTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/AST/ASTDumperTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/AST/ASTExprTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/AST/ASTImporterGenericRedeclTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/AST/ASTImporterODRStrategiesTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/AST/AttrTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/AST/ConceptPrinterTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/AST/DataCollectionTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/AST/DeclPrinterTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/AST/NamedDeclPrinterTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/AST/QualTypeNamesTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/AST/SizelessTypesTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/AST/StmtPrinterTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/AST/TemplateNameTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/AST/TypePrinterTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/CodeGen/TBAAMetadataTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Driver/ToolChainTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Interpreter/InterpreterTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Lex/NoTrivialPPDirectiveTracerTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Lex/PPConditionalDirectiveRecordTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Lex/PPDependencyDirectivesTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Lex/PPMemoryAllocationsTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Lex/PPCallbacksTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Lex/ModuleDeclStateTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Parse/ParseHLSLRootSignatureTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/ScalableStaticAnalysisFramework/Registries/MockSerializationFormat.cpp
    ${LLVM_REPO_DIR}/clang/unittests/StaticAnalyzer/BlockEntranceCallbackTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/StaticAnalyzer/BugReportInterestingnessTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/StaticAnalyzer/CallDescriptionTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/StaticAnalyzer/CallEventTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/StaticAnalyzer/ConflictingEvalCallsTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/StaticAnalyzer/ExprEngineVisitTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/StaticAnalyzer/FalsePositiveRefutationBRVisitorTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/StaticAnalyzer/MemRegionDescriptiveNameTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/StaticAnalyzer/NoStateChangeFuncVisitorTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/StaticAnalyzer/ObjcBug-124477.cpp
    ${LLVM_REPO_DIR}/clang/unittests/StaticAnalyzer/RegisterCustomCheckersTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/StaticAnalyzer/SValSimplifyerTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/StaticAnalyzer/SValTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/StaticAnalyzer/TestReturnValueUnderConstruction.cpp
    ${LLVM_REPO_DIR}/clang/unittests/StaticAnalyzer/UnsignedStatDemo.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Tooling/DiagnosticsYamlTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Tooling/RecursiveASTVisitorTests/CallbacksBinaryOperator.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Tooling/RecursiveASTVisitorTests/CallbacksCallExpr.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Tooling/RecursiveASTVisitorTests/CallbacksCompoundAssignOperator.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Tooling/RecursiveASTVisitorTests/CallbacksLeaf.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Tooling/RecursiveASTVisitorTests/CallbacksUnaryOperator.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Tooling/SourceCodeBuildersTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Tooling/SourceCodeTest.cpp
    ${LLVM_REPO_DIR}/clang/unittests/Tooling/StencilTest.cpp
    TARGET_DIRECTORY AllClangUnitTests
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET FormatTests)
  # From clang/unittests/Format/CMakeLists.txt (target: FormatTests)
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang/unittests/Format/FormatTestJS.cpp
    TARGET_DIRECTORY FormatTests
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET clang-tblgen)
  # From clang/utils/TableGen/CMakeLists.txt (target: clang-tblgen)
  # MveEmitter.cpp and NeonEmitter.cpp define large anonymous-namespace classes
  # that conflict with each other in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/clang/utils/TableGen/NeonEmitter.cpp
    TARGET_DIRECTORY clang-tblgen
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

# ======================================================================
# flang
# ======================================================================

if(TARGET FortranEvaluate)
  # From flang/lib/Evaluate/CMakeLists.txt (target: FortranEvaluate)
  # fold-implementation.h contains explicit instantiation definitions; merging
  # any two of these files in a unity TU produces duplicate instantiations.
  # formatting.cpp also has explicit instantiations that duplicate those.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/flang/lib/Evaluate/fold-character.cpp
    ${LLVM_REPO_DIR}/flang/lib/Evaluate/fold-complex.cpp
    ${LLVM_REPO_DIR}/flang/lib/Evaluate/fold-integer.cpp
    ${LLVM_REPO_DIR}/flang/lib/Evaluate/fold-logical.cpp
    ${LLVM_REPO_DIR}/flang/lib/Evaluate/fold-real.cpp
    ${LLVM_REPO_DIR}/flang/lib/Evaluate/fold.cpp
    ${LLVM_REPO_DIR}/flang/lib/Evaluate/intrinsics-library.cpp
    ${LLVM_REPO_DIR}/flang/lib/Evaluate/formatting.cpp
    TARGET_DIRECTORY FortranEvaluate
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET FortranLower)
  # From flang/lib/Lower/CMakeLists.txt (target: FortranLower)
  # ConvertCall.cpp calls 'isInWhereMaskedExpression' which becomes ambiguous
  # with another overload brought in from other files in the unity TU.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/flang/lib/Lower/ConvertCall.cpp
    TARGET_DIRECTORY FortranLower
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET FIRBuilder)
  # From flang/lib/Optimizer/Builder/CMakeLists.txt (target: FIRBuilder)
  # Execute.cpp defines a static 'isAbsent' that conflicts with another
  # definition when merged in a unity TU.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/flang/lib/Optimizer/Builder/Runtime/Execute.cpp
    TARGET_DIRECTORY FIRBuilder
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET FIRCodeGen)
  # From flang/lib/Optimizer/CodeGen/CMakeLists.txt (target: FIRCodeGen)
  # FIROpPatterns.cpp and PreCGRewrite.cpp each redefine local helpers
  # (getTypeDescFieldId, DeclareOpConversion) that clash in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/flang/lib/Optimizer/CodeGen/FIROpPatterns.cpp
    ${LLVM_REPO_DIR}/flang/lib/Optimizer/CodeGen/PreCGRewrite.cpp
    TARGET_DIRECTORY FIRCodeGen
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET HLFIRTransforms)
  # From flang/lib/Optimizer/HLFIR/Transforms/CMakeLists.txt (target: HLFIRTransforms)
  # ConvertToFIR.cpp redefines AssignOpConversion and NoReassocOpConversion,
  # ScheduleOrderedAssignments.cpp redefines isForallIndex, and
  # SimplifyHLFIRIntrinsics.cpp redefines CmpCharOpConversion/IndexOpConversion.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/flang/lib/Optimizer/HLFIR/Transforms/ConvertToFIR.cpp
    ${LLVM_REPO_DIR}/flang/lib/Optimizer/HLFIR/Transforms/ScheduleOrderedAssignments.cpp
    ${LLVM_REPO_DIR}/flang/lib/Optimizer/HLFIR/Transforms/SimplifyHLFIRIntrinsics.cpp
    TARGET_DIRECTORY HLFIRTransforms
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET FIRTransforms)
  # From flang/lib/Optimizer/Transforms/CMakeLists.txt (target: FIRTransforms)
  # AnnotateConstant.cpp has 'impl' ambiguity from tablegen pass base classes.
  # LoopInvariantCodeMotion.cpp has fir::AliasAnalysis vs mlir::AliasAnalysis
  # ambiguity. CUFOpConversionLate.cpp redefines createConvertOp.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/flang/lib/Optimizer/Transforms/AnnotateConstant.cpp
    ${LLVM_REPO_DIR}/flang/lib/Optimizer/Transforms/ConvertComplexPow.cpp
    ${LLVM_REPO_DIR}/flang/lib/Optimizer/Transforms/LoopInvariantCodeMotion.cpp
    ${LLVM_REPO_DIR}/flang/lib/Optimizer/Transforms/CUDA/CUFOpConversionLate.cpp
    TARGET_DIRECTORY FIRTransforms
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET f18-parse-demo)
  # From flang/tools/f18-parse-demo/CMakeLists.txt (target: f18-parse-demo)
  # stub-evaluate.cpp uses struct/class mismatches with other TU definitions
  # that trigger -Werror,-Wmismatched-tags in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/flang/tools/f18-parse-demo/stub-evaluate.cpp
    TARGET_DIRECTORY f18-parse-demo
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

# ======================================================================
# llvm
# ======================================================================

if(TARGET LLVMCodeGen)
  set_source_files_properties(
    # From llvm/lib/CodeGen/CMakeLists.txt (target: LLVMCodeGen)
    # STATISTIC and static function redefinitions across register allocation
    # and scheduling files conflict in unity builds.
    ${LLVM_REPO_DIR}/llvm/lib/CodeGen/LiveIntervalCalc.cpp
    ${LLVM_REPO_DIR}/llvm/lib/CodeGen/MLRegAllocPriorityAdvisor.cpp
    ${LLVM_REPO_DIR}/llvm/lib/CodeGen/MachineOutliner.cpp
    ${LLVM_REPO_DIR}/llvm/lib/CodeGen/PostRASchedulerList.cpp
    ${LLVM_REPO_DIR}/llvm/lib/CodeGen/RegAllocPriorityAdvisor.cpp
    # From llvm/lib/CodeGen/CMakeLists.txt (target: LLVMCodeGen)
    # Conflicting STATISTIC variables and static functions with other
    # register allocation files.
    ${LLVM_REPO_DIR}/llvm/lib/CodeGen/MLRegAllocEvictAdvisor.cpp
    # From llvm/lib/CodeGen/CMakeLists.txt (target: LLVMCodeGen)
    # Conflicting static cl::opt variables with other CodeGen files.
    ${LLVM_REPO_DIR}/llvm/lib/CodeGen/ModuloSchedule.cpp
    # From llvm/lib/CodeGen/CMakeLists.txt (target: LLVMCodeGen)
    # Anonymous namespace struct redefinitions conflict in unity builds.
    ${LLVM_REPO_DIR}/llvm/lib/CodeGen/PatchableFunction.cpp
    TARGET_DIRECTORY LLVMCodeGen
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMGlobalISel)
  # From llvm/lib/CodeGen/GlobalISel/CMakeLists.txt (target: LLVMGlobalISel)
  # Conflicting static helper functions and STATISTIC variables.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/CodeGen/GlobalISel/IRTranslator.cpp
    ${LLVM_REPO_DIR}/llvm/lib/CodeGen/GlobalISel/LegacyLegalizerInfo.cpp
    TARGET_DIRECTORY LLVMGlobalISel
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMMIRParser)
  # From llvm/lib/CodeGen/MIRParser/CMakeLists.txt (target: LLVMMIRParser)
  # Static functions and anonymous namespace classes conflict in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/CodeGen/MIRParser/MIParser.cpp
    TARGET_DIRECTORY LLVMMIRParser
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMSelectionDAG)
  set_source_files_properties(
    # From llvm/lib/CodeGen/SelectionDAG/CMakeLists.txt (target: LLVMSelectionDAG)
    # STATISTIC variable redefinitions (NumBacktracks, NumUnfolds, etc.)
    # and static function conflicts.
    ${LLVM_REPO_DIR}/llvm/lib/CodeGen/SelectionDAG/ScheduleDAGRRList.cpp
    ${LLVM_REPO_DIR}/llvm/lib/CodeGen/SelectionDAG/SelectionDAG.cpp
    # From llvm/lib/CodeGen/SelectionDAG/CMakeLists.txt (target: LLVMSelectionDAG)
    # Large template instantiation code and static helper conflicts.
    ${LLVM_REPO_DIR}/llvm/lib/CodeGen/SelectionDAG/TargetLowering.cpp
    TARGET_DIRECTORY LLVMSelectionDAG
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMJITLink)
  # From llvm/lib/ExecutionEngine/JITLink/CMakeLists.txt (target: LLVMJITLink)
  # Anonymous namespace function redefinitions (buildTables_ELF_x86*).
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/ExecutionEngine/JITLink/ELF_x86.cpp
    ${LLVM_REPO_DIR}/llvm/lib/ExecutionEngine/JITLink/ELF_x86_64.cpp
    TARGET_DIRECTORY LLVMJITLink
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMOrcJIT)
  # From llvm/lib/ExecutionEngine/Orc/CMakeLists.txt (target: LLVMOrcJIT)
  # Static function and class definition conflicts in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/ExecutionEngine/Orc/ELFNixPlatform.cpp
    ${LLVM_REPO_DIR}/llvm/lib/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.cpp
    TARGET_DIRECTORY LLVMOrcJIT
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMRuntimeDyld)
  # From llvm/lib/ExecutionEngine/RuntimeDyld/CMakeLists.txt (target: LLVMRuntimeDyld)
  # Static relocation helper functions (or32le, getBits, etc.) conflict.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/ExecutionEngine/RuntimeDyld/RuntimeDyldELF.cpp
    TARGET_DIRECTORY LLVMRuntimeDyld
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMObjCopy)
  # From llvm/lib/ObjCopy/CMakeLists.txt (target: LLVMObjCopy)
  # ELF, MachO, COFF, and wasm ObjCopy implementations define identically-named
  # types (Section, Object, SectionPred) in different namespaces that conflict
  # when combined via "using namespace" in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/ObjCopy/MachO/MachOObjcopy.cpp
    ${LLVM_REPO_DIR}/llvm/lib/ObjCopy/MachO/MachOReader.cpp
    TARGET_DIRECTORY LLVMObjCopy
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMObjectYAML)
  # From llvm/lib/ObjectYAML/CMakeLists.txt (target: LLVMObjectYAML)
  # Conflicting 'using namespace' declarations (llvm::codeview,
  # llvm::CodeViewYAML) cause ambiguity.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/ObjectYAML/CodeViewYAMLTypes.cpp
    ${LLVM_REPO_DIR}/llvm/lib/ObjectYAML/MinidumpEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/lib/ObjectYAML/MinidumpYAML.cpp
    TARGET_DIRECTORY LLVMObjectYAML
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMProfileData)
  set_source_files_properties(
    # From llvm/lib/ProfileData/CMakeLists.txt (target: LLVMProfileData)
    # InstrProfReader.cpp and InstrProfWriter.cpp have ambiguous 'Summary' due
    # to namespace conflicts in unity builds.
    ${LLVM_REPO_DIR}/llvm/lib/ProfileData/InstrProfReader.cpp
    ${LLVM_REPO_DIR}/llvm/lib/ProfileData/InstrProfWriter.cpp
    ${LLVM_REPO_DIR}/llvm/lib/ProfileData/SampleProf.cpp
    # From llvm/lib/ProfileData/CMakeLists.txt (target: LLVMProfileData)
    # 'using namespace sampleprof' causes ambiguity with other files.
    ${LLVM_REPO_DIR}/llvm/lib/ProfileData/SampleProfReader.cpp
    TARGET_DIRECTORY LLVMProfileData
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMSupport)
  # From llvm/lib/Support/CMakeLists.txt (target: LLVMSupport)
  # error: expected unqualified-id 672 |   while (const auto *IN = dyn_cast<RopePieceBTreeInterior>(N))
  # error: expected unqualified-id 511 |     for (unsigned i = 0, e = std::min(ThisWords, RHSWords); i != e; ++i)
  if(WIN32)
    set_source_files_properties(
      ${LLVM_REPO_DIR}/llvm/lib/Support/RewriteRope.cpp
      ${LLVM_REPO_DIR}/llvm/lib/Support/ManagedStatic.cpp
      TARGET_DIRECTORY LLVMSupport
      PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
  endif()
endif()

if(TARGET AArch64)
  set_source_files_properties(
    # From llvm/lib/Target/AArch64/CMakeLists.txt (target: AArch64)
    # Anonymous namespace classes and static function conflicts.
    ${LLVM_REPO_DIR}/llvm/lib/Target/AArch64/AArch64AdvSIMDScalarPass.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/AArch64/SVEIntrinsicOpts.cpp
    # From llvm/lib/Target/AArch64/CMakeLists.txt (target: AArch64)
    # Large ISel lowering file with extensive static helper conflicts.
    ${LLVM_REPO_DIR}/llvm/lib/Target/AArch64/AArch64ISelLowering.cpp
    # From llvm/lib/Target/AArch64/CMakeLists.txt (target: AArch64)
    ${LLVM_REPO_DIR}/llvm/lib/Target/AArch64/SMEABIPass.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/AArch64/GISel/AArch64CallLowering.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/AArch64/GISel/AArch64InstructionSelector.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/AArch64/GISel/AArch64LegalizerInfo.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/AArch64/GISel/AArch64O0PreLegalizerCombiner.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/AArch64/GISel/AArch64PostLegalizerCombiner.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/AArch64/GISel/AArch64PostLegalizerLowering.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/AArch64/GISel/AArch64PreLegalizerCombiner.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/AArch64/GISel/AArch64RegisterBankInfo.cpp
    TARGET_DIRECTORY AArch64
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET AMDGPU)
  set_source_files_properties(
    # From llvm/lib/Target/AMDGPU/CMakeLists.txt (target: AMDGPU)
    # Conflicting DAG patterns and static functions across AMDGPU files.
    ${LLVM_REPO_DIR}/llvm/lib/Target/AMDGPU/AMDGPUISelDAGToDAG.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/AMDGPU/AMDGPUInstCombineIntrinsic.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/AMDGPU/AMDGPURegisterBankInfo.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/AMDGPU/AMDGPUSwLowerLDS.cpp
    # From llvm/lib/Target/AMDGPU/CMakeLists.txt (target: AMDGPU)
    # Generated GISel combiner code with conflicting rule symbols.
    ${LLVM_REPO_DIR}/llvm/lib/Target/AMDGPU/AMDGPUInstructionSelector.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/AMDGPU/AMDGPUPostLegalizerCombiner.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/AMDGPU/AMDGPUPreLegalizerCombiner.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/AMDGPU/AMDGPURegBankCombiner.cpp
    TARGET_DIRECTORY AMDGPU
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMARMDesc)
  # From llvm/lib/Target/ARM/MCTargetDesc/CMakeLists.txt (target: LLVMARMDesc)
  # Static ELF relocation helper functions conflict in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/Target/ARM/MCTargetDesc/ARMELFObjectWriter.cpp
    TARGET_DIRECTORY LLVMARMDesc
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET Hexagon)
  # From llvm/lib/Target/Hexagon/CMakeLists.txt (target: Hexagon)
  # Static analysis functions conflict with other Hexagon pass files.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/Target/Hexagon/HexagonOptimizeSZextends.cpp
    TARGET_DIRECTORY Hexagon
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MSP430)
  # From llvm/lib/Target/MSP430/CMakeLists.txt (target: MSP430)
  # Static instruction lowering helpers conflict in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/Target/MSP430/MSP430MCInstLower.cpp
    TARGET_DIRECTORY MSP430
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMNVPTXDesc)
  # From llvm/lib/Target/NVPTX/MCTargetDesc/CMakeLists.txt (target: LLVMNVPTXDesc)
  # Static table and helper function redefinitions.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/Target/NVPTX/MCTargetDesc/NVPTXMCTargetDesc.cpp
    TARGET_DIRECTORY LLVMNVPTXDesc
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET RISCV)
  set_source_files_properties(
    # From llvm/lib/Target/RISCV/CMakeLists.txt (target: RISCV)
    # Static analysis functions conflict across RISCV passes.
    ${LLVM_REPO_DIR}/llvm/lib/Target/RISCV/RISCVInsertWriteVXRM.cpp
    # From llvm/lib/Target/RISCV/CMakeLists.txt (target: RISCV)
    # Generated combiner code with conflicting rule symbols.
    ${LLVM_REPO_DIR}/llvm/lib/Target/RISCV/RISCVO0PreLegalizerCombiner.cpp
    # From llvm/lib/Target/RISCV/CMakeLists.txt (target: RISCV)
    # Generated GISel combiner code with overlapping rule identifiers.
    ${LLVM_REPO_DIR}/llvm/lib/Target/RISCV/GISel/RISCVO0PreLegalizerCombiner.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/RISCV/GISel/RISCVPreLegalizerCombiner.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/RISCV/GISel/RISCVPostLegalizerCombiner.cpp
    TARGET_DIRECTORY RISCV
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET SPIRV)
  # From llvm/lib/Target/SPIRV/CMakeLists.txt (target: SPIRV)
  # Conflicting intrinsic handling and static functions across SPIRV files.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/Target/SPIRV/SPIRVBuiltins.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/SPIRV/SPIRVEmitIntrinsics.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/SPIRV/SPIRVModuleAnalysis.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/SPIRV/SPIRVPreLegalizerCombiner.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/SPIRV/SPIRVRegisterBankInfo.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/SPIRV/SPIRVRegisterInfo.cpp
    TARGET_DIRECTORY SPIRV
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMSPIRVDesc)
  # From llvm/lib/Target/SPIRV/MCTargetDesc/CMakeLists.txt (target: LLVMSPIRVDesc)
  # Static encoding tables and helper function redefinitions.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/Target/SPIRV/MCTargetDesc/SPIRVBaseInfo.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/SPIRV/MCTargetDesc/SPIRVMCCodeEmitter.cpp
    TARGET_DIRECTORY LLVMSPIRVDesc
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET WebAssembly)
  set_source_files_properties(
    # From llvm/lib/Target/WebAssembly/CMakeLists.txt (target: WebAssembly)
    # Static CFG transformation functions conflict with other WebAssembly files.
    ${LLVM_REPO_DIR}/llvm/lib/Target/WebAssembly/WebAssemblyFixIrreducibleControlFlow.cpp
    # From llvm/lib/Target/WebAssembly/CMakeLists.txt (target: WebAssembly)
    # Generated combiner code with conflicting rule symbols.
    ${LLVM_REPO_DIR}/llvm/lib/Target/WebAssembly/WebAssemblyPreLegalizerCombiner.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/WebAssembly/WebAssemblyPostLegalizerCombiner.cpp
    # From llvm/lib/Target/WebAssembly/CMakeLists.txt (target: WebAssembly)
    # Generated GISel combiner code with conflicting symbols.
    ${LLVM_REPO_DIR}/llvm/lib/Target/WebAssembly/GISel/WebAssemblyPreLegalizerCombiner.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/WebAssembly/GISel/WebAssemblyPostLegalizerCombiner.cpp
    TARGET_DIRECTORY WebAssembly
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET X86)
  set_source_files_properties(
    # From llvm/lib/Target/X86/CMakeLists.txt (target: X86)
    # Large ISel lowering and static helper function conflicts across X86 files.
    ${LLVM_REPO_DIR}/llvm/lib/Target/X86/X86FastTileConfig.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/X86/X86FixupVectorConstants.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/X86/X86ISelLowering.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/X86/X86LoadValueInjectionRetHardening.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/X86/X86LowerAMXIntrinsics.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/X86/X86SpeculativeExecutionSideEffectSuppression.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/X86/X86WinEHState.cpp
    # From llvm/lib/Target/X86/CMakeLists.txt (target: X86)
    # Static instruction pattern matching and encoding table conflicts.
    ${LLVM_REPO_DIR}/llvm/lib/Target/X86/X86CompressEVEX.cpp
    # From llvm/lib/Target/X86/CMakeLists.txt (target: X86)
    # Generated GISel combiner code with overlapping rule symbols.
    ${LLVM_REPO_DIR}/llvm/lib/Target/X86/GISel/X86PreLegalizerCombiner.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Target/X86/GISel/X86PostLegalizerCombiner.cpp
    TARGET_DIRECTORY X86
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET XCore)
  set_source_files_properties(
    # From llvm/lib/Target/XCore/CMakeLists.txt (target: XCore)
    # Static instruction information and helper function conflicts.
    ${LLVM_REPO_DIR}/llvm/lib/Target/XCore/XCoreInstrInfo.cpp
    # From llvm/lib/Target/XCore/CMakeLists.txt (target: XCore)
    # Static register information and allocation helper conflicts.
    ${LLVM_REPO_DIR}/llvm/lib/Target/XCore/XCoreRegisterInfo.cpp
    TARGET_DIRECTORY XCore
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMCoroutines)
  # From llvm/lib/Transforms/Coroutines/CMakeLists.txt (target: LLVMCoroutines)
  # Anonymous namespace Lowerer classes and static coroutine helpers conflict.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/Transforms/Coroutines/CoroEarly.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Transforms/Coroutines/CoroElide.cpp
    TARGET_DIRECTORY LLVMCoroutines
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMInstrumentation)
  # From llvm/lib/Transforms/Instrumentation/CMakeLists.txt (target: LLVMInstrumentation)
  # Static instrumentation functions and 'using namespace' cause ambiguity.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/Transforms/Instrumentation/HWAddressSanitizer.cpp
    TARGET_DIRECTORY LLVMInstrumentation
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMObjCARCOpts)
  # From llvm/lib/Transforms/ObjCARC/CMakeLists.txt (target: LLVMObjCARCOpts)
  # Static optimization functions conflict with other ObjCARC files.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/Transforms/ObjCARC/ObjCARCContract.cpp
    TARGET_DIRECTORY LLVMObjCARCOpts
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMScalarOpts)
  # From llvm/lib/Transforms/Scalar/CMakeLists.txt (target: LLVMScalarOpts)
  # STATISTIC variables and static analysis function redefinitions.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/Transforms/Scalar/BDCE.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Transforms/Scalar/ConstraintElimination.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Transforms/Scalar/GVNSink.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Transforms/Scalar/LoopInterchange.cpp
    TARGET_DIRECTORY LLVMScalarOpts
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMTransformUtils)
  set_source_files_properties(
    # From llvm/lib/Transforms/Utils/CMakeLists.txt (target: LLVMTransformUtils)
    # Static lowering functions conflict with other Utils files.
    ${LLVM_REPO_DIR}/llvm/lib/Transforms/Utils/LowerGlobalDtors.cpp
    # From llvm/lib/Transforms/Utils/CMakeLists.txt (target: LLVMTransformUtils)
    # Static exception handling transformation functions conflict.
    ${LLVM_REPO_DIR}/llvm/lib/Transforms/Utils/LowerInvoke.cpp
    TARGET_DIRECTORY LLVMTransformUtils
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMVectorize)
  # From llvm/lib/Transforms/Vectorize/CMakeLists.txt (target: LLVMVectorize)
  # VPlan files define conflicting STATISTIC variables, anonymous namespace
  # classes, and static pattern matching functions in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/Transforms/Vectorize/LoopVectorize.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Transforms/Vectorize/VPlan.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Transforms/Vectorize/VPlanAnalysis.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Transforms/Vectorize/VPlanConstruction.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Transforms/Vectorize/VPlanPredicator.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Transforms/Vectorize/VPlanRecipes.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Transforms/Vectorize/VPlanTransforms.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Transforms/Vectorize/VPlanUnroll.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Transforms/Vectorize/VPlanUtils.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Transforms/Vectorize/VPlanVerifier.cpp
    TARGET_DIRECTORY LLVMVectorize
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET dsymutil)
  # From llvm/tools/dsymutil/CMakeLists.txt (target: dsymutil)
  # CFBundle.cpp includes ObjC headers that conflict with MachO headers
  # from other files in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/tools/dsymutil/CFBundle.cpp
    ${LLVM_REPO_DIR}/llvm/tools/dsymutil/RelocationMap.cpp
    TARGET_DIRECTORY dsymutil
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET llvm-c-test)
  # From llvm/tools/llvm-c-test/CMakeLists.txt (target: llvm-c-test)
  # calc.c and disassemble.c both define handle_line, causing
  # redefinition errors in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/tools/llvm-c-test/calc.c
    ${LLVM_REPO_DIR}/llvm/tools/llvm-c-test/disassemble.c
    ${LLVM_REPO_DIR}/llvm/tools/llvm-c-test/diagnostic.c
    ${LLVM_REPO_DIR}/llvm/tools/llvm-c-test/module.c
    TARGET_DIRECTORY llvm-c-test
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET llvm-jitlink)
  # From llvm/tools/llvm-jitlink/CMakeLists.txt (target: llvm-jitlink)
  # ELF, MachO, and COFF jitlink files define identically-named static functions
  # (getFirstRelocationEdge, etc.) that conflict in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/tools/llvm-jitlink/llvm-jitlink-elf.cpp
    ${LLVM_REPO_DIR}/llvm/tools/llvm-jitlink/llvm-jitlink-macho.cpp
    TARGET_DIRECTORY llvm-jitlink
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET llvm-objdump)
  # From llvm/tools/llvm-objdump/CMakeLists.txt (target: llvm-objdump)
  # MachODump.cpp defines UnwindInfo that conflicts with other dump files'
  # types when combined in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/tools/llvm-objdump/MachODump.cpp
    TARGET_DIRECTORY llvm-objdump
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET llvm-pdbutil)
  # From llvm/tools/llvm-pdbutil/CMakeLists.txt (target: llvm-pdbutil)
  # DumpOutputStyle.cpp defines printHeader which conflicts with identically-named
  # functions from other output style files in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/tools/llvm-pdbutil/DumpOutputStyle.cpp
    TARGET_DIRECTORY llvm-pdbutil
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET llvm-readobj)
  # From llvm/tools/llvm-readobj/CMakeLists.txt (target: llvm-readobj)
  # ELFDumper.cpp defines createError and checkHashTable that conflict with
  # identically-named functions from other dumper files in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/tools/llvm-readobj/ELFDumper.cpp
    TARGET_DIRECTORY llvm-readobj
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET llvm-reduce)
  # From llvm/tools/llvm-reduce/CMakeLists.txt (target: llvm-reduce)
  # ReduceOperandsSkip.cpp defines shouldReduceOperand which conflicts with
  # identically-named functions from other delta files in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/tools/llvm-reduce/deltas/ReduceOperandsSkip.cpp
    TARGET_DIRECTORY llvm-reduce
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET llvm-min-tblgen)
  set_source_files_properties(
    # From llvm/utils/TableGen/CMakeLists.txt (target: llvm-min-tblgen)
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/AsmWriterEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/CTagsEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/CallingConvEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/CodeEmitterGen.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/CompressInstEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/DAGISelEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/DXILEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/DisassemblerEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/ExegesisEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/FastISelEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/GlobalISelCombinerEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/GlobalISelEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/InstrDocsEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/MacroFusionPredicatorEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/OptionParserEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/OptionRSTEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/PseudoLoweringEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/RegisterBankEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/RegisterInfoEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/SearchableTableEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/SubtargetEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/X86FoldTablesEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/X86InstrMappingEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/X86MnemonicTables.cpp
    # From llvm/utils/TableGen/CMakeLists.txt (target: llvm-min-tblgen)
    # Static DFA state tables and code generation helper conflicts.
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/DFAPacketizerEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/InstrInfoEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/SDNodeInfoEmitter.cpp
    TARGET_DIRECTORY llvm-min-tblgen
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMBitReader)
  # From llvm/lib/Bitcode/Reader/CMakeLists.txt (target: LLVMBitReader)
  # Both files define a static 'error' function that conflicts in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/Bitcode/Reader/BitcodeReader.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Bitcode/Reader/MetadataLoader.cpp
    TARGET_DIRECTORY LLVMBitReader
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMDebugInfoCodeView)
  # From llvm/lib/DebugInfo/CodeView/CMakeLists.txt (target: LLVMDebugInfoCodeView)
  # These files define static functions/arrays with identical names (stabilize,
  # LeafTypeNames, getLeafTypeName) that conflict in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/DebugInfo/CodeView/AppendingTypeTableBuilder.cpp
    ${LLVM_REPO_DIR}/llvm/lib/DebugInfo/CodeView/GlobalTypeTableBuilder.cpp
    ${LLVM_REPO_DIR}/llvm/lib/DebugInfo/CodeView/MergingTypeTableBuilder.cpp
    ${LLVM_REPO_DIR}/llvm/lib/DebugInfo/CodeView/TypeDumpVisitor.cpp
    ${LLVM_REPO_DIR}/llvm/lib/DebugInfo/CodeView/TypeRecordMapping.cpp
    TARGET_DIRECTORY LLVMDebugInfoCodeView
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMDebugInfoGSYM)
  # From llvm/lib/DebugInfo/GSYM/CMakeLists.txt (target: LLVMDebugInfoGSYM)
  # FunctionInfo.cpp defines an enum value 'InlineInfo' that conflicts with
  # the struct InlineInfo in InlineInfo.cpp in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/DebugInfo/GSYM/FunctionInfo.cpp
    ${LLVM_REPO_DIR}/llvm/lib/DebugInfo/GSYM/InlineInfo.cpp
    TARGET_DIRECTORY LLVMDebugInfoGSYM
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMDemangle)
  # From llvm/lib/Demangle/CMakeLists.txt (target: LLVMDemangle)
  # These demangler files each define classes/type aliases named 'Demangler' and
  # use "using namespace" directives that bring conflicting symbols (Node,
  # Qualifiers, etc.) into scope from different namespaces, which cannot coexist
  # in a single translation unit.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/Demangle/ItaniumDemangle.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Demangle/MicrosoftDemangle.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Demangle/RustDemangle.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Demangle/DLangDemangle.cpp
    TARGET_DIRECTORY LLVMDemangle
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMFrontendHLSL)
  # From llvm/lib/Frontend/HLSL/CMakeLists.txt (target: LLVMFrontendHLSL)
  # RootSignatureMetadata.cpp redefines 'OverloadedVisit' from
  # HLSLRootSignature.cpp in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/Frontend/HLSL/HLSLRootSignature.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Frontend/HLSL/RootSignatureMetadata.cpp
    TARGET_DIRECTORY LLVMFrontendHLSL
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMMC)
  # From llvm/lib/MC/CMakeLists.txt (target: LLVMMC)
  # These object writer files each define 'isDwoSection' in anonymous namespaces,
  # which conflicts in unity builds. MCAsmInfoELF/Wasm both define 'printName'.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/MC/ELFObjectWriter.cpp
    ${LLVM_REPO_DIR}/llvm/lib/MC/MCAsmInfoELF.cpp
    ${LLVM_REPO_DIR}/llvm/lib/MC/MCAsmInfoWasm.cpp
    ${LLVM_REPO_DIR}/llvm/lib/MC/WasmObjectWriter.cpp
    ${LLVM_REPO_DIR}/llvm/lib/MC/WinCOFFObjectWriter.cpp
    TARGET_DIRECTORY LLVMMC
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMObject)
  # From llvm/lib/Object/CMakeLists.txt (target: LLVMObject)
  # MachOUniversal.cpp/MachOObjectFile.cpp: ambiguous 'malformedError'
  # ModuleSymbolTable.cpp: 'Module' conflicts with llvm::minidump::Module
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/Object/MachOObjectFile.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Object/MachOUniversal.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Object/ModuleSymbolTable.cpp
    ${LLVM_REPO_DIR}/llvm/lib/Object/OffloadBinary.cpp
    TARGET_DIRECTORY LLVMObject
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMTargetParser)
  # From llvm/lib/TargetParser/CMakeLists.txt (target: LLVMTargetParser)
  # Unity build conflicts:
  # - CSKYTargetParser.cpp: redefines 'stripNegationPrefix' from ARMTargetParser
  # - X86TargetParser.cpp: 'FeatureBitset' ambiguous due to namespace conflicts
  # - RISCVTargetParser.cpp: incomplete type 'CPUInfo[]' due to include ordering
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/TargetParser/CSKYTargetParser.cpp
    ${LLVM_REPO_DIR}/llvm/lib/TargetParser/RISCVTargetParser.cpp
    ${LLVM_REPO_DIR}/llvm/lib/TargetParser/X86TargetParser.cpp
    TARGET_DIRECTORY LLVMTargetParser
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMTextAPI)
  # From llvm/lib/TextAPI/CMakeLists.txt (target: LLVMTextAPI)
  # RecordsSlice.cpp has ambiguous 'Target' due to namespace conflicts in
  # unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/lib/TextAPI/RecordsSlice.cpp
    TARGET_DIRECTORY LLVMTextAPI
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET LLVMTableGenBasic)
  # From llvm/utils/TableGen/Basic/CMakeLists.txt (target: LLVMTableGenBasic)
  # These emitter files each define a static TableGen::Emitter::Opt[Class]
  # variable named 'X' (and sometimes 'Y') at file scope for self-registration,
  # which conflicts in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/Basic/ARMTargetDefEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/Basic/Attributes.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/Basic/DirectiveEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/Basic/IntrinsicEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/Basic/RISCVTargetDefEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/Basic/RuntimeLibcallsEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/Basic/TargetFeaturesEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/Basic/TargetLibraryInfoEmitter.cpp
    ${LLVM_REPO_DIR}/llvm/utils/TableGen/Basic/VTEmitter.cpp
    TARGET_DIRECTORY LLVMTableGenBasic
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

# ======================================================================
# mlir
# ======================================================================

if(TARGET toyc-ch3)
  # From mlir/examples/toy/Ch3/CMakeLists.txt (target: toyc-ch3)
  # mlir/ToyCombine.cpp references toy:: names that become ambiguous when
  # combined with other mlir/ files in a unity translation unit.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/examples/toy/Ch3/mlir/ToyCombine.cpp
    TARGET_DIRECTORY toyc-ch3
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRSCFToGPU)
  # From mlir/lib/Conversion/SCFToGPU/CMakeLists.txt (target: MLIRSCFToGPU)
  # Conversion patterns and static helpers conflict via 'using namespace mlir'.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/lib/Conversion/SCFToGPU/SCFToGPUPass.cpp
    TARGET_DIRECTORY MLIRSCFToGPU
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRDebug)
  # From mlir/lib/Debug/CMakeLists.txt (target: MLIRDebug)
  # Static hook registration and context handling code conflicts.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/lib/Debug/DebuggerExecutionContextHook.cpp
    TARGET_DIRECTORY MLIRDebug
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRAffineTransforms)
  # From mlir/lib/Dialect/Affine/Transforms/CMakeLists.txt (target: MLIRAffineTransforms)
  # Anonymous namespace struct redefinitions (LowerDelinearizeIndexOps, etc.).
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/Affine/Transforms/AffineExpandIndexOpsAsAffine.cpp
    TARGET_DIRECTORY MLIRAffineTransforms
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRArithTransforms)
  # From mlir/lib/Dialect/Arith/Transforms/CMakeLists.txt (target: MLIRArithTransforms)
  # Static interface implementations conflict via 'using namespace mlir'.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.cpp
    TARGET_DIRECTORY MLIRArithTransforms
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRBufferizationTransforms)
  # From mlir/lib/Dialect/Bufferization/Transforms/CMakeLists.txt (target: MLIRBufferizationTransforms)
  # Conflicting analysis pattern definitions.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.cpp
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/Bufferization/Transforms/OwnershipBasedBufferDeallocation.cpp
    TARGET_DIRECTORY MLIRBufferizationTransforms
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRLinalgTransforms)
  # From mlir/lib/Dialect/Linalg/Transforms/CMakeLists.txt (target: MLIRLinalgTransforms)
  # Anonymous namespace pattern rewriter struct redefinitions.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/Linalg/Transforms/DecomposeLinalgOps.cpp
    TARGET_DIRECTORY MLIRLinalgTransforms
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRMemRefTransforms)
  # From mlir/lib/Dialect/MemRef/Transforms/CMakeLists.txt (target: MLIRMemRefTransforms)
  # Static type emulation pattern and helper function conflicts.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/MemRef/Transforms/EmulateNarrowType.cpp
    TARGET_DIRECTORY MLIRMemRefTransforms
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIROpenACCDialect)
  # From mlir/lib/Dialect/OpenACC/IR/CMakeLists.txt (target: MLIROpenACCDialect)
  # Static code generation and type conversion function conflicts.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/OpenACC/IR/OpenACCCG.cpp
    TARGET_DIRECTORY MLIROpenACCDialect
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRPtrDialect)
  # From mlir/lib/Dialect/Ptr/IR/CMakeLists.txt (target: MLIRPtrDialect)
  # Static pointer type handling conflicts via 'using namespace mlir'.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/Ptr/IR/PtrTypes.cpp
    TARGET_DIRECTORY MLIRPtrDialect
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRSparseTensorTransforms)
  # From mlir/lib/Dialect/SparseTensor/Transforms/CMakeLists.txt (target: MLIRSparseTensorTransforms)
  # Conflicting sparse tensor analysis and lowering pattern definitions.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/SparseTensor/Transforms/SparseTensorConversion.cpp
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/SparseTensor/Transforms/Sparsification.cpp
    TARGET_DIRECTORY MLIRSparseTensorTransforms
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRVectorTransforms)
  # From mlir/lib/Dialect/Vector/Transforms/CMakeLists.txt (target: MLIRVectorTransforms)
  # Anonymous namespace rewrite pattern redefinitions.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/Vector/Transforms/VectorInsertExtractStridedSliceRewritePatterns.cpp
    TARGET_DIRECTORY MLIRVectorTransforms
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRPDLLCodeGen)
  # From mlir/lib/Tools/PDLL/CodeGen/CMakeLists.txt (target: MLIRPDLLCodeGen)
  # Static PDLL code generation helpers and symbol registration conflicts.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/lib/Tools/PDLL/CodeGen/MLIRGen.cpp
    TARGET_DIRECTORY MLIRPDLLCodeGen
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRLspServerLib)
  # From mlir/lib/Tools/mlir-lsp-server/CMakeLists.txt (target: MLIRLspServerLib)
  # Static LSP protocol handler and request processing conflicts.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/lib/Tools/mlir-lsp-server/MLIRServer.cpp
    TARGET_DIRECTORY MLIRLspServerLib
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET TableGenLspServerLib)
  # From mlir/lib/Tools/tblgen-lsp-server/CMakeLists.txt (target: TableGenLspServerLib)
  # Static TableGen LSP handler conflicts.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/lib/Tools/tblgen-lsp-server/TableGenServer.cpp
    TARGET_DIRECTORY TableGenLspServerLib
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRTransformUtils)
  # From mlir/lib/Transforms/Utils/CMakeLists.txt (target: MLIRTransformUtils)
  # Anonymous namespace debugging infrastructure and static helper conflicts.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp
    TARGET_DIRECTORY MLIRTransformUtils
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRTestAnalysis)
  # From mlir/test/lib/Analysis/CMakeLists.txt (target: MLIRTestAnalysis)
  # Static test infrastructure redefinitions across analysis test files.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/test/lib/Analysis/TestMemRefDependenceCheck.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Analysis/TestSlice.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Analysis/DataFlow/TestDeadCodeAnalysis.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Analysis/DataFlow/TestDenseBackwardDataFlowAnalysis.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Analysis/DataFlow/TestDenseForwardDataFlowAnalysis.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Analysis/DataFlow/TestLivenessAnalysis.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Analysis/DataFlow/TestSparseBackwardDataFlowAnalysis.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Analysis/DataFlow/TestStridedMetadataRangeAnalysis.cpp
    TARGET_DIRECTORY MLIRTestAnalysis
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRTestFuncToLLVM)
  # From mlir/test/lib/Conversion/FuncToLLVM/CMakeLists.txt (target: MLIRTestFuncToLLVM)
  # Static test pass and conversion pattern conflicts.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/test/lib/Conversion/FuncToLLVM/TestConvertFuncOp.cpp
    TARGET_DIRECTORY MLIRTestFuncToLLVM
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRAffineTransformsTestPasses)
  set_source_files_properties(
    # From mlir/test/lib/Dialect/Affine/CMakeLists.txt (target: MLIRAffineTransformsTestPasses)
    # Static vectorization test utility conflicts.
    ${LLVM_REPO_DIR}/mlir/test/lib/Dialect/Affine/TestVectorizationUtils.cpp
    # From mlir/test/lib/Dialect/Affine/CMakeLists.txt (target: MLIRAffineTransformsTestPasses)
    # Static value bound analysis test infrastructure conflicts.
    ${LLVM_REPO_DIR}/mlir/test/lib/Dialect/Affine/TestReifyValueBounds.cpp
    # From mlir/test/lib/Dialect/Affine/CMakeLists.txt (target: MLIRAffineTransformsTestPasses)
    # Static analysis test harness redefinitions.
    ${LLVM_REPO_DIR}/mlir/test/lib/Dialect/Affine/TestAccessAnalysis.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Dialect/Affine/TestAffineLoopUnswitching.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Dialect/Affine/TestDecomposeAffineOps.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Dialect/Affine/TestLoopFusion.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Dialect/Affine/TestLoopPermutation.cpp
    TARGET_DIRECTORY MLIRAffineTransformsTestPasses
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRBufferizationTestPasses)
  # From mlir/test/lib/Dialect/Bufferization/CMakeLists.txt (target: MLIRBufferizationTestPasses)
  # Static test pattern implementation conflicts.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/test/lib/Dialect/Bufferization/TestTensorLikeAndBufferLike.cpp
    TARGET_DIRECTORY MLIRBufferizationTestPasses
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRSCFTestPasses)
  # From mlir/test/lib/Dialect/SCF/CMakeLists.txt (target: MLIRSCFTestPasses)
  # Static loop unrolling test pattern conflicts.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/test/lib/Dialect/SCF/TestParallelLoopUnrolling.cpp
    TARGET_DIRECTORY MLIRSCFTestPasses
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRTestTransformDialect)
  # From mlir/test/lib/Dialect/Transform/CMakeLists.txt (target: MLIRTestTransformDialect)
  # Conflicting static test interpreter definitions.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/test/lib/Dialect/Transform/TestTransformDialectInterpreter.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Dialect/Transform/TestTransformStateExtension.cpp
    TARGET_DIRECTORY MLIRTestTransformDialect
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRTestIR)
  set_source_files_properties(
    # From mlir/test/lib/IR/CMakeLists.txt (target: MLIRTestIR)
    # Static dominance analysis test infrastructure conflicts.
    ${LLVM_REPO_DIR}/mlir/test/lib/IR/TestDominance.cpp
    # From mlir/test/lib/IR/CMakeLists.txt (target: MLIRTestIR)
    ${LLVM_REPO_DIR}/mlir/test/lib/IR/TestBuiltinAttributeInterfaces.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/IR/TestBuiltinDistinctAttributes.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/IR/TestBytecodeRoundtrip.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/IR/TestClone.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/IR/TestLazyLoading.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/IR/TestRegions.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/IR/TestSideEffects.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/IR/TestSymbolUses.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/IR/TestTypes.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/IR/TestVisitorsGeneric.cpp
    # From mlir/test/lib/IR/CMakeLists.txt (target: MLIRTestIR)
    # Static interface testing utility conflicts.
    ${LLVM_REPO_DIR}/mlir/test/lib/IR/TestInterfaces.cpp
    TARGET_DIRECTORY MLIRTestIR
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRTestPass)
  # From mlir/test/lib/Pass/CMakeLists.txt (target: MLIRTestPass)
  # Static pass management test code conflicts.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/test/lib/Pass/TestPassManager.cpp
    TARGET_DIRECTORY MLIRTestPass
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRTestTransforms)
  # From mlir/test/lib/Transforms/CMakeLists.txt (target: MLIRTestTransforms)
  # Static transform test utility redefinitions.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/test/lib/Transforms/TestCommutativityUtils.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Transforms/TestDialectConversion.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Transforms/TestInlining.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Transforms/TestInliningCallback.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Transforms/TestMakeIsolatedFromAbove.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Transforms/TestTransformDialectExtension.cpp
    TARGET_DIRECTORY MLIRTestTransforms
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRPassTests)
  # From mlir/unittests/Pass/CMakeLists.txt (target: MLIRPassTests)
  # Static pass management unit test infrastructure conflicts.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/unittests/Pass/PassManagerTest.cpp
    TARGET_DIRECTORY MLIRPassTests
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRLLVMCommonConversion)
  # From mlir/lib/Conversion/LLVMCommon/CMakeLists.txt (target: MLIRLLVMCommonConversion)
  # 'using namespace mlir' causes Type and other core type ambiguity.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/lib/Conversion/LLVMCommon/StructBuilder.cpp
    ${LLVM_REPO_DIR}/mlir/lib/Conversion/LLVMCommon/TypeConverter.cpp
    TARGET_DIRECTORY MLIRLLVMCommonConversion
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRShapeToStandard)
  # From mlir/lib/Conversion/ShapeToStandard/CMakeLists.txt (target: MLIRShapeToStandard)
  # Conflicting pattern rewriters and helper function redefinitions.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/lib/Conversion/ShapeToStandard/ShapeToStandard.cpp
    ${LLVM_REPO_DIR}/mlir/lib/Conversion/ShapeToStandard/ConvertShapeConstraints.cpp
    TARGET_DIRECTORY MLIRShapeToStandard
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRSPIRVDialect)
  # From mlir/lib/Dialect/SPIRV/IR/CMakeLists.txt (target: MLIRSPIRVDialect)
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/SPIRV/IR/ArmGraphOps.cpp
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/SPIRV/IR/AtomicOps.cpp
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/SPIRV/IR/CastOps.cpp
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/SPIRV/IR/ControlFlowOps.cpp
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/SPIRV/IR/CooperativeMatrixOps.cpp
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/SPIRV/IR/DotProductOps.cpp
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/SPIRV/IR/GroupOps.cpp
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/SPIRV/IR/MemoryOps.cpp
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/SPIRV/IR/SPIRVCanonicalization.cpp
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/SPIRV/IR/SPIRVDialect.cpp
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/SPIRV/IR/SPIRVGLCanonicalization.cpp
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/SPIRV/IR/SPIRVOpDefinition.cpp
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/SPIRV/IR/SPIRVOps.cpp
    ${LLVM_REPO_DIR}/mlir/lib/Dialect/SPIRV/IR/SPIRVParsingUtils.cpp
    TARGET_DIRECTORY MLIRSPIRVDialect
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRIR)
  # From mlir/lib/IR/CMakeLists.txt (target: MLIRIR)
  # SymbolTable.cpp has ambiguous 'function_ref', 'SetVector', 'detail', 'impl'
  # due to 'using namespace llvm;' from other files in the unity batch.
  # Types.cpp has ambiguous 'Type' (mlir::Type vs llvm::Type).
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/lib/IR/SymbolTable.cpp
    ${LLVM_REPO_DIR}/mlir/lib/IR/TypeRange.cpp
    ${LLVM_REPO_DIR}/mlir/lib/IR/TypeUtilities.cpp
    ${LLVM_REPO_DIR}/mlir/lib/IR/Types.cpp
    TARGET_DIRECTORY MLIRIR
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRTableGen)
  # From mlir/lib/TableGen/CMakeLists.txt (target: MLIRTableGen)
  # CodeGenHelpers.cpp has "using namespace llvm;" which makes DenseMapInfo
  # ambiguous in Constraint.cpp's DenseMapInfo<Constraint> implementations
  # when compiled together in a unity build.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/lib/TableGen/Constraint.cpp
    TARGET_DIRECTORY MLIRTableGen
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRTestDialect)
  # From mlir/test/lib/Dialect/Test/CMakeLists.txt (target: MLIRTestDialect)
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/test/lib/Dialect/Test/TestAttributes.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Dialect/Test/TestDialect.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Dialect/Test/TestDialectInterfaces.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Dialect/Test/TestInterfaces.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Dialect/Test/TestOpDefs.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Dialect/Test/TestOps.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Dialect/Test/TestOpsSyntax.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Dialect/Test/TestPatterns.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Dialect/Test/TestTraits.cpp
    ${LLVM_REPO_DIR}/mlir/test/lib/Dialect/Test/TestTypes.cpp
    TARGET_DIRECTORY MLIRTestDialect
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET mlir-tblgen)
  # From mlir/tools/mlir-tblgen/CMakeLists.txt (target: mlir-tblgen)
  # Unity build conflicts:
  # - BytecodeDialectGen.cpp/DialectGen.cpp: both define static 'dialectGenCat'
  # - PassGen.cpp/PassCAPIGen.cpp: both define 'passGenCat', 'groupName',
  #   'passGroupRegistrationCode', 'fileHeader'
  # - SPIRVUtilsGen.cpp/TosaUtilsGen.cpp: both define 'Availability' class and
  #   related functions in anonymous namespaces
  # - LLVMIRIntrinsicGen.cpp, OpDocGen.cpp, OpPythonBindingGen.cpp, RewriterGen.cpp:
  #   'using namespace llvm;' from other files makes DenseMap, Type, StringSwitch,
  #   SetVector ambiguous with mlir::tblgen equivalents
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/tools/mlir-tblgen/BytecodeDialectGen.cpp
    ${LLVM_REPO_DIR}/mlir/tools/mlir-tblgen/DialectGen.cpp
    ${LLVM_REPO_DIR}/mlir/tools/mlir-tblgen/LLVMIRIntrinsicGen.cpp
    ${LLVM_REPO_DIR}/mlir/tools/mlir-tblgen/OpDocGen.cpp
    ${LLVM_REPO_DIR}/mlir/tools/mlir-tblgen/OpFormatGen.cpp
    ${LLVM_REPO_DIR}/mlir/tools/mlir-tblgen/OpPythonBindingGen.cpp
    ${LLVM_REPO_DIR}/mlir/tools/mlir-tblgen/PassCAPIGen.cpp
    ${LLVM_REPO_DIR}/mlir/tools/mlir-tblgen/PassGen.cpp
    ${LLVM_REPO_DIR}/mlir/tools/mlir-tblgen/RewriterGen.cpp
    ${LLVM_REPO_DIR}/mlir/tools/mlir-tblgen/SPIRVUtilsGen.cpp
    ${LLVM_REPO_DIR}/mlir/tools/mlir-tblgen/TosaUtilsGen.cpp
    TARGET_DIRECTORY mlir-tblgen
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRExecutionEngineTests)
  # From mlir/unittests/ExecutionEngine/CMakeLists.txt (target: MLIRExecutionEngineTests)
  # These files include CRunnerUtils.h/RunnerUtils.h which defines
  # UnrankedMemRefType that conflicts with mlir::UnrankedMemRefType in unity builds.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/unittests/ExecutionEngine/DynamicMemRef.cpp
    ${LLVM_REPO_DIR}/mlir/unittests/ExecutionEngine/StridedMemRef.cpp
    ${LLVM_REPO_DIR}/mlir/unittests/ExecutionEngine/Invoke.cpp
    TARGET_DIRECTORY MLIRExecutionEngineTests
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

if(TARGET MLIRIRTests)
  # From mlir/unittests/IR/CMakeLists.txt (target: MLIRIRTests)
  # Static test helper function redefinitions across unit tests.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/mlir/unittests/IR/AffineMapTest.cpp
    ${LLVM_REPO_DIR}/mlir/unittests/IR/OpPropertiesTest.cpp
    ${LLVM_REPO_DIR}/mlir/unittests/IR/TypeAttrNamesTest.cpp
    ${LLVM_REPO_DIR}/mlir/unittests/IR/ValueTest.cpp
    TARGET_DIRECTORY MLIRIRTests
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

# ======================================================================
# polly
# ======================================================================

if(TARGET Polly)
  # From polly/lib/CMakeLists.txt (target: Polly)
  # ISLOStream.h has no include guard, so its inline operator<< definitions get
  # redefined when multiple .cpp files including it are in the same unity TU.
  # Several Analysis/ and CodeGen/ files also share STATISTIC variable names.
  set_source_files_properties(
    ${LLVM_REPO_DIR}/polly/lib/Analysis/ScopDetection.cpp
    ${LLVM_REPO_DIR}/polly/lib/Analysis/ScopInfo.cpp
    ${LLVM_REPO_DIR}/polly/lib/Analysis/ScopBuilder.cpp
    ${LLVM_REPO_DIR}/polly/lib/Analysis/PruneUnprofitable.cpp
    ${LLVM_REPO_DIR}/polly/lib/CodeGen/IslAst.cpp
    ${LLVM_REPO_DIR}/polly/lib/CodeGen/CodeGeneration.cpp
    ${LLVM_REPO_DIR}/polly/lib/Transform/ScheduleOptimizer.cpp
    ${LLVM_REPO_DIR}/polly/lib/Transform/FlattenSchedule.cpp
    ${LLVM_REPO_DIR}/polly/lib/Transform/FlattenAlgo.cpp
    ${LLVM_REPO_DIR}/polly/lib/Transform/ForwardOpTree.cpp
    ${LLVM_REPO_DIR}/polly/lib/Transform/DeLICM.cpp
    ${LLVM_REPO_DIR}/polly/lib/Transform/Simplify.cpp
    ${LLVM_REPO_DIR}/polly/lib/Transform/MatmulOptimizer.cpp
    TARGET_DIRECTORY Polly
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
endif()

# ======================================================================
# Targets with UNITY_BUILD OFF
# ======================================================================
# Some directories have pervasive unity build conflicts (duplicate symbols,
# ObjC++ files, using-namespace collisions, etc.) that make per-file
# SKIP_UNITY_BUILD_INCLUSION impractical. Disable unity build entirely
# for these targets.

# Helper: recursively disable unity build for all targets under a directory.
function(_disable_unity_build_in_dir dir)
  get_property(_targets DIRECTORY ${dir} PROPERTY BUILDSYSTEM_TARGETS)
  foreach(_t ${_targets})
    set_target_properties(${_t} PROPERTIES UNITY_BUILD OFF)
  endforeach()
  get_property(_subdirs DIRECTORY ${dir} PROPERTY SUBDIRECTORIES)
  foreach(_sd ${_subdirs})
    _disable_unity_build_in_dir(${_sd})
  endforeach()
endfunction()

# -- Explicit targets (small directories with 1-2 targets) --
foreach(_target
    # clang-tools-extra/clang-tidy (30+ check module targets with AST_MATCHER conflicts)
    clangTidy
    clangTidyAbseilModule
    clangTidyAlteraModule
    clangTidyAndroidModule
    clangTidyBoostModule
    clangTidyBugproneModule
    clangTidyCERTModule
    clangTidyConcurrencyModule
    clangTidyCppCoreGuidelinesModule
    clangTidyCustomModule
    clangTidyDarwinModule
    clangTidyFuchsiaModule
    clangTidyGoogleModule
    clangTidyHICPPModule
    clangTidyLinuxKernelModule
    clangTidyLLVMLibcModule
    clangTidyLLVMModule
    clangTidyMain
    clangTidyMiscModule
    clangTidyModernizeModule
    clangTidyMPIModule
    clangTidyObjCModule
    clangTidyOpenMPModule
    clangTidyPerformanceModule
    clangTidyPlugin
    clangTidyPortabilityModule
    clangTidyReadabilityModule
    clangTidyUtils
    clangTidyZirconModule
    # clang-tools-extra/clangd/fuzzer
    clangd-fuzzer
    # clang/lib/Driver
    clangDriver
    # clang/unittests
    AllClangUnitTests
    BasicTests
    FormatTests
    SemaTests
    ClangScalableAnalysisTests
    ClangReplInterpreterTests
    CIRUnitTests
    # llvm/lib/CAS
    LLVMCAS
    # llvm/lib/DebugInfo/LogicalView
    LLVMDebugInfoLogicalView
    # llvm/tools/llvm-remarkutil
    llvm-remarkutil
    # llvm/tools/llvm-xray
    llvm-xray
    # mlir/examples/toy
    toyc-ch4
    toyc-ch5
    toyc-ch6
    toyc-ch7)
  if(TARGET ${_target})
    set_target_properties(${_target} PROPERTIES UNITY_BUILD OFF)
  endif()
  # Also handle object library targets (obj.X) created by add_*_library macros
  if(TARGET obj.${_target})
    set_target_properties(obj.${_target} PROPERTIES UNITY_BUILD OFF)
  endif()
endforeach()

# -- Directory-based (directories with many subtargets) --
# lld: COFF/ELF/MachO/MinGW/Wasm targets + unittests
if(TARGET lldCommon)
  _disable_unity_build_in_dir(${LLVM_REPO_DIR}/lld)
endif()

# lldb: 180+ targets with pervasive "using namespace lldb" conflicts
if(TARGET lldb-server OR TARGET liblldb)
  _disable_unity_build_in_dir(${LLVM_REPO_DIR}/lldb)
endif()

# libclc: dynamically-generated targets
if(TARGET libclc)
  _disable_unity_build_in_dir(${LLVM_REPO_DIR}/libclc)
endif()

# llvm/unittests: 75+ unittest targets
if(TARGET UnitTests)
  _disable_unity_build_in_dir(${LLVM_SRC_DIR}/unittests)
endif()
