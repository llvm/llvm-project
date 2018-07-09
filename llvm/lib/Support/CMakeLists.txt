set(system_libs)
if ( LLVM_ENABLE_ZLIB AND HAVE_LIBZ )
  set(system_libs ${system_libs} ${ZLIB_LIBRARIES})
endif()
if( MSVC OR MINGW )
  # libuuid required for FOLDERID_Profile usage in lib/Support/Windows/Path.inc.
  # advapi32 required for CryptAcquireContextW in lib/Support/Windows/Path.inc.
  set(system_libs ${system_libs} psapi shell32 ole32 uuid advapi32)
elseif( CMAKE_HOST_UNIX )
  if( HAVE_LIBRT )
    set(system_libs ${system_libs} rt)
  endif()
  if( HAVE_LIBDL )
    set(system_libs ${system_libs} ${CMAKE_DL_LIBS})
  endif()
  if( HAVE_BACKTRACE AND NOT "${Backtrace_LIBRARIES}" STREQUAL "" )
    # On BSDs, CMake returns a fully qualified path to the backtrace library.
    # We need to remove the path and the 'lib' prefix, to make it look like a
    # regular short library name, suitable for appending to a -l link flag.
    get_filename_component(Backtrace_LIBFILE ${Backtrace_LIBRARIES} NAME_WE)
    STRING(REGEX REPLACE "^lib" "" Backtrace_LIBFILE ${Backtrace_LIBFILE})
    set(system_libs ${system_libs} ${Backtrace_LIBFILE})
  endif()
  if(LLVM_ENABLE_TERMINFO)
    if(HAVE_TERMINFO)
      set(system_libs ${system_libs} ${TERMINFO_LIBS})
    endif()
  endif()
  if( LLVM_ENABLE_THREADS AND HAVE_LIBATOMIC )
    set(system_libs ${system_libs} atomic)
  endif()
  set(system_libs ${system_libs} ${LLVM_PTHREAD_LIB})
  if( UNIX AND NOT (BEOS OR HAIKU) )
    set(system_libs ${system_libs} m)
  endif()
  if( FUCHSIA )
    set(system_libs ${system_libs} zircon)
  endif()
endif( MSVC OR MINGW )

add_llvm_library(LLVMSupport
  AMDGPUMetadata.cpp
  APFloat.cpp
  APInt.cpp
  APSInt.cpp
  ARMBuildAttrs.cpp
  ARMAttributeParser.cpp
  ARMWinEH.cpp
  Allocator.cpp
  BinaryStreamError.cpp
  BinaryStreamReader.cpp
  BinaryStreamRef.cpp
  BinaryStreamWriter.cpp
  BlockFrequency.cpp
  BranchProbability.cpp
  CachePruning.cpp
  circular_raw_ostream.cpp
  Chrono.cpp
  COM.cpp
  CodeGenCoverage.cpp
  CommandLine.cpp
  Compression.cpp
  ConvertUTF.cpp
  ConvertUTFWrapper.cpp
  CrashRecoveryContext.cpp
  DataExtractor.cpp
  Debug.cpp
  DebugCounter.cpp
  DeltaAlgorithm.cpp
  DAGDeltaAlgorithm.cpp
  DJB.cpp
  Error.cpp
  ErrorHandling.cpp
  FileUtilities.cpp
  FileOutputBuffer.cpp
  FoldingSet.cpp
  FormattedStream.cpp
  FormatVariadic.cpp
  GlobPattern.cpp
  GraphWriter.cpp
  Hashing.cpp
  InitLLVM.cpp
  IntEqClasses.cpp
  IntervalMap.cpp
  JamCRC.cpp
  JSON.cpp
  KnownBits.cpp
  LEB128.cpp
  LineIterator.cpp
  Locale.cpp
  LockFileManager.cpp
  LowLevelType.cpp
  ManagedStatic.cpp
  MathExtras.cpp
  MemoryBuffer.cpp
  MD5.cpp
  NativeFormatting.cpp
  Options.cpp
  Parallel.cpp
  PluginLoader.cpp
  PrettyStackTrace.cpp
  RandomNumberGenerator.cpp
  Regex.cpp
  ScaledNumber.cpp
  ScopedPrinter.cpp
  SHA1.cpp
  SmallPtrSet.cpp
  SmallVector.cpp
  SourceMgr.cpp
  SpecialCaseList.cpp
  Statistic.cpp
  StringExtras.cpp
  StringMap.cpp
  StringPool.cpp
  StringSaver.cpp
  StringRef.cpp
  SystemUtils.cpp
  TarWriter.cpp
  TargetParser.cpp
  ThreadPool.cpp
  Timer.cpp
  ToolOutputFile.cpp
  TrigramIndex.cpp
  Triple.cpp
  Twine.cpp
  Unicode.cpp
  UnicodeCaseFold.cpp
  VersionTuple.cpp
  WithColor.cpp
  YAMLParser.cpp
  YAMLTraits.cpp
  raw_os_ostream.cpp
  raw_ostream.cpp
  regcomp.c
  regerror.c
  regexec.c
  regfree.c
  regstrlcpy.c
  xxhash.cpp

# System
  Atomic.cpp
  DynamicLibrary.cpp
  Errno.cpp
  Host.cpp
  Memory.cpp
  Mutex.cpp
  Path.cpp
  Process.cpp
  Program.cpp
  RWMutex.cpp
  Signals.cpp
  TargetRegistry.cpp
  ThreadLocal.cpp
  Threading.cpp
  Valgrind.cpp
  Watchdog.cpp

  ADDITIONAL_HEADER_DIRS
  Unix
  Windows
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ADT
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Support
  ${Backtrace_INCLUDE_DIRS}
  LINK_LIBS ${system_libs}
  )

set_property(TARGET LLVMSupport PROPERTY LLVM_SYSTEM_LIBS "${system_libs}")
