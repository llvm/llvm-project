set(LLVM_LINK_COMPONENTS
  Support
  TargetParser
  )

add_llvm_unittest(SupportTests
  AddressRangeTest.cpp
  AlignmentTest.cpp
  AlignOfTest.cpp
  AllocatorTest.cpp
  ARMAttributeParser.cpp
  ArrayRecyclerTest.cpp
  Base64Test.cpp
  BinaryStreamTest.cpp
  BLAKE3Test.cpp
  BlockFrequencyTest.cpp
  BalancedPartitioningTest.cpp
  BranchProbabilityTest.cpp
  CachePruningTest.cpp
  CrashRecoveryTest.cpp
  Caching.cpp
  Casting.cpp
  CheckedArithmeticTest.cpp
  Chrono.cpp
  CommandLineTest.cpp
  CompressionTest.cpp
  ConvertEBCDICTest.cpp
  ConvertUTFTest.cpp
  CRCTest.cpp
  CSKYAttributeParserTest.cpp
  DataExtractorTest.cpp
  DebugCounterTest.cpp
  DebugTest.cpp
  DebugLogTest.cpp
  DivisionByConstantTest.cpp
  DJBTest.cpp
  EndianStreamTest.cpp
  EndianTest.cpp
  ELFAttributeParserTest.cpp
  ErrnoTest.cpp
  ErrorOrTest.cpp
  ErrorTest.cpp
  ExponentialBackoffTest.cpp
  ExtensibleRTTITest.cpp
  FileCollectorTest.cpp
  FileOutputBufferTest.cpp
  FormatVariadicTest.cpp
  FSUniqueIDTest.cpp
  GenericDomTreeTest.cpp
  GlobPatternTest.cpp
  HashBuilderTest.cpp
  IndexedAccessorTest.cpp
  InstructionCostTest.cpp
  InterleavedRangeTest.cpp
  JSONTest.cpp
  KnownBitsTest.cpp
  LEB128Test.cpp
  LineIteratorTest.cpp
  LockFileManagerTest.cpp
  MatchersTest.cpp
  MD5Test.cpp
  ManagedStatic.cpp
  MathExtrasTest.cpp
  MemoryBufferRefTest.cpp
  MemoryBufferTest.cpp
  MemoryTest.cpp
  MustacheTest.cpp
  ModRefTest.cpp
  NativeFormatTests.cpp
  OptimizedStructLayoutTest.cpp
  ParallelTest.cpp
  Path.cpp
  PerThreadBumpPtrAllocatorTest.cpp
  ProcessTest.cpp
  ProgramTest.cpp
  ProgramStackTest.cpp
  RecyclerTest.cpp
  RegexTest.cpp
  ReverseIterationTest.cpp
  ReplaceFileTest.cpp
  RISCVAttributeParserTest.cpp
  ScaledNumberTest.cpp
  ScopedPrinterTest.cpp
  SHA256.cpp
  SignalsTest.cpp
  SipHashTest.cpp
  SourceMgrTest.cpp
  SpecialCaseListTest.cpp
  SuffixTreeTest.cpp
  SwapByteOrderTest.cpp
  TarWriterTest.cpp
  TextEncodingTest.cpp
  ThreadPool.cpp
  ThreadSafeAllocatorTest.cpp
  Threading.cpp
  TimerTest.cpp
  TimeProfilerTest.cpp
  ToolOutputFileTest.cpp
  TypeNameTest.cpp
  TypeSizeTest.cpp
  TypeTraitsTest.cpp
  TrailingObjectsTest.cpp
  UnicodeTest.cpp
  UTCTimeTest.cpp
  VersionTupleTest.cpp
  VirtualFileSystemTest.cpp
  VirtualOutputBackendTest.cpp
  VirtualOutputBackendsTest.cpp
  VirtualOutputConfigTest.cpp
  VirtualOutputFileTest.cpp
  WithColorTest.cpp
  YAMLIOTest.cpp
  YAMLParserTest.cpp
  buffer_ostream_test.cpp
  formatted_raw_ostream_test.cpp
  raw_fd_stream_test.cpp
  raw_ostream_test.cpp
  raw_ostream_proxy_test.cpp
  raw_pwrite_stream_test.cpp
  raw_sha1_ostream_test.cpp
  raw_socket_stream_test.cpp
  xxhashTest.cpp

  DEPENDS
  intrinsics_gen
  )

add_subdirectory(LSP)

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

if(NOT LLVM_INTEGRATED_CRT_ALLOC)
  # The test doesn't pass when using a custom allocator, PR47881.
  add_subdirectory(DynamicLibrary)
endif()

add_subdirectory(CommandLineInit)
