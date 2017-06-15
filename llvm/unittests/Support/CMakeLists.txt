set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_unittest(SupportTests
  AlignOfTest.cpp
  AllocatorTest.cpp
  ARMAttributeParser.cpp
  ArrayRecyclerTest.cpp
  BinaryStreamTest.cpp
  BlockFrequencyTest.cpp
  BranchProbabilityTest.cpp
  CachePruningTest.cpp
  CrashRecoveryTest.cpp
  Casting.cpp
  Chrono.cpp
  CommandLineTest.cpp
  CompressionTest.cpp
  ConvertUTFTest.cpp
  DataExtractorTest.cpp
  DebugTest.cpp
  EndianStreamTest.cpp
  EndianTest.cpp
  ErrorOrTest.cpp
  ErrorTest.cpp
  FileOutputBufferTest.cpp
  FormatVariadicTest.cpp
  GlobPatternTest.cpp
  Host.cpp
  LEB128Test.cpp
  LineIteratorTest.cpp
  LockFileManagerTest.cpp
  MD5Test.cpp
  ManagedStatic.cpp
  MathExtrasTest.cpp
  MemoryBufferTest.cpp
  MemoryTest.cpp
  NativeFormatTests.cpp
  ParallelTest.cpp
  Path.cpp
  ProcessTest.cpp
  ProgramTest.cpp
  RegexTest.cpp
  ReplaceFileTest.cpp
  ScaledNumberTest.cpp
  SourceMgrTest.cpp
  SpecialCaseListTest.cpp
  StringPool.cpp
  SwapByteOrderTest.cpp
  TarWriterTest.cpp
  TargetParserTest.cpp
  ThreadLocalTest.cpp
  ThreadPool.cpp
  Threading.cpp
  TimerTest.cpp
  TypeNameTest.cpp
  TrailingObjectsTest.cpp
  TrigramIndexTest.cpp
  UnicodeTest.cpp
  YAMLIOTest.cpp
  YAMLParserTest.cpp
  formatted_raw_ostream_test.cpp
  raw_ostream_test.cpp
  raw_pwrite_stream_test.cpp
  raw_sha1_ostream_test.cpp
  xxhashTest.cpp
  )

# Disable all warning for AlignOfTest.cpp,
# as it does things intentionally, and there is no reliable way of
# disabling all warnings for all the compilers by using pragmas.
set_source_files_properties(AlignOfTest.cpp PROPERTIES COMPILE_FLAGS -w)

# ManagedStatic.cpp uses <pthread>.
target_link_libraries(SupportTests LLVMTestingSupport ${LLVM_PTHREAD_LIB})

add_subdirectory(DynamicLibrary)
