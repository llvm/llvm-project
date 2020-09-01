set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_unittest(SupportTests
  AlignmentTest.cpp
  AlignOfTest.cpp
  AllocatorTest.cpp
  AnnotationsTest.cpp
  ARMAttributeParser.cpp
  ArrayRecyclerTest.cpp
  Base64Test.cpp
  BinaryStreamTest.cpp
  BlockFrequencyTest.cpp
  BranchProbabilityTest.cpp
  CachePruningTest.cpp
  CrashRecoveryTest.cpp
  Casting.cpp
  CheckedArithmeticTest.cpp
  Chrono.cpp
  CommandLineTest.cpp
  CompressionTest.cpp
  ConvertUTFTest.cpp
  CRCTest.cpp
  DataExtractorTest.cpp
  DebugTest.cpp
  DebugCounterTest.cpp
  DJBTest.cpp
  EndianStreamTest.cpp
  EndianTest.cpp
  ELFAttributeParserTest.cpp
  ErrnoTest.cpp
  ErrorOrTest.cpp
  ErrorTest.cpp
  ExtensibleRTTITest.cpp
  FileCollectorTest.cpp
  FileOutputBufferTest.cpp
  FileUtilitiesTest.cpp
  FormatVariadicTest.cpp
  GlobPatternTest.cpp
  Host.cpp
  IndexedAccessorTest.cpp
  ItaniumManglingCanonicalizerTest.cpp
  JSONTest.cpp
  KnownBitsTest.cpp
  LEB128Test.cpp
  LineIteratorTest.cpp
  LockFileManagerTest.cpp
  MatchersTest.cpp
  MD5Test.cpp
  ManagedStatic.cpp
  MathExtrasTest.cpp
  MemoryBufferTest.cpp
  MemoryTest.cpp
  NativeFormatTests.cpp
  OptimizedStructLayoutTest.cpp
  ParallelTest.cpp
  Path.cpp
  ProcessTest.cpp
  ProgramTest.cpp
  RegexTest.cpp
  ReverseIterationTest.cpp
  ReplaceFileTest.cpp
  RISCVAttributeParserTest.cpp
  ScaledNumberTest.cpp
  SourceMgrTest.cpp
  SpecialCaseListTest.cpp
  SuffixTreeTest.cpp
  SwapByteOrderTest.cpp
  SymbolRemappingReaderTest.cpp
  TarWriterTest.cpp
  TargetParserTest.cpp
  TaskQueueTest.cpp
  ThreadLocalTest.cpp
  ThreadPool.cpp
  Threading.cpp
  TimerTest.cpp
  ToolOutputFileTest.cpp
  TypeNameTest.cpp
  TypeTraitsTest.cpp
  TrailingObjectsTest.cpp
  TrigramIndexTest.cpp
  UnicodeTest.cpp
  VersionTupleTest.cpp
  VirtualFileSystemTest.cpp
  WithColorTest.cpp
  YAMLIOTest.cpp
  YAMLParserTest.cpp
  formatted_raw_ostream_test.cpp
  raw_ostream_test.cpp
  raw_pwrite_stream_test.cpp
  raw_sha1_ostream_test.cpp
  xxhashTest.cpp
  )

target_link_libraries(SupportTests PRIVATE LLVMTestingSupport)

# Disable all warning for AlignOfTest.cpp,
# as it does things intentionally, and there is no reliable way of
# disabling all warnings for all the compilers by using pragmas.
# Don't disable on MSVC, because all incriminated warnings are already disabled
# in source; and because we would otherwise see this warning:
#   cl : Command line warning D9025: overriding '/W4' with '/w'
if(NOT MSVC)
  set_source_files_properties(AlignOfTest.cpp PROPERTIES COMPILE_FLAGS -w)
endif()
if(MSVC)
  if( CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 19.14 )
    # Since VS2017 15.8, the following snippet: Failed<CustomSubError>()
    # generates a warning:
    # \svn\llvm\utils\unittest\googlemock\include\gmock\gmock-matchers.h(186):
    #   warning C5046: 'testing::MatcherInterface<T>::~MatcherInterface': Symbol involving type with internal linkage not defined
    set_source_files_properties(ErrorTest.cpp PROPERTIES COMPILE_FLAGS -wd5046)
  endif()
endif()

# ManagedStatic.cpp uses <pthread>.
target_link_libraries(SupportTests PRIVATE LLVMTestingSupport ${LLVM_PTHREAD_LIB})

add_subdirectory(DynamicLibrary)
