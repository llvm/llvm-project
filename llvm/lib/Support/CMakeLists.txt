set(system_libs)
if( MSVC OR MINGW )
  # libuuid required for FOLDERID_Profile usage in lib/Support/Windows/Path.inc.
  set(system_libs ${system_libs} psapi shell32 ole32 uuid)
elseif( CMAKE_HOST_UNIX )
  if( HAVE_LIBRT )
    set(system_libs ${system_libs} rt)
  endif()
  if( HAVE_LIBDL )
    set(system_libs ${system_libs} ${CMAKE_DL_LIBS})
  endif()
  if(LLVM_ENABLE_TERMINFO)
    if(HAVE_TERMINFO)
      set(system_libs ${system_libs} ${TERMINFO_LIBS})
    endif()
  endif()
  if( LLVM_ENABLE_THREADS AND HAVE_LIBATOMIC )
    set(system_libs ${system_libs} atomic)
  endif()
  set(system_libs ${system_libs} ${PTHREAD_LIB})
  if ( LLVM_ENABLE_ZLIB AND HAVE_LIBZ )
    set(system_libs ${system_libs} z)
  endif()
  if( UNIX AND NOT (BEOS OR HAIKU) )
    set(system_libs ${system_libs} m)
  endif()
endif( MSVC OR MINGW )

add_llvm_library(LLVMSupport
  APFloat.cpp
  APInt.cpp
  APSInt.cpp
  ARMBuildAttrs.cpp
  ARMWinEH.cpp
  Allocator.cpp
  BlockFrequency.cpp
  BranchProbability.cpp
  CachePruning.cpp
  circular_raw_ostream.cpp
  COM.cpp
  CommandLine.cpp
  Compression.cpp
  ConvertUTF.c
  ConvertUTFWrapper.cpp
  CrashRecoveryContext.cpp
  DataExtractor.cpp
  DataStream.cpp
  Debug.cpp
  DeltaAlgorithm.cpp
  DAGDeltaAlgorithm.cpp
  Dwarf.cpp
  Error.cpp
  ErrorHandling.cpp
  FileUtilities.cpp
  FileOutputBuffer.cpp
  FoldingSet.cpp
  FormattedStream.cpp
  GraphWriter.cpp
  Hashing.cpp
  IntEqClasses.cpp
  IntervalMap.cpp
  IntrusiveRefCntPtr.cpp
  JamCRC.cpp
  LEB128.cpp
  LineIterator.cpp
  Locale.cpp
  LockFileManager.cpp
  ManagedStatic.cpp
  MathExtras.cpp
  MemoryBuffer.cpp
  MemoryObject.cpp
  MD5.cpp
  Options.cpp
  PluginLoader.cpp
  PrettyStackTrace.cpp
  RandomNumberGenerator.cpp
  Regex.cpp
  ScaledNumber.cpp
  SHA1.cpp
  SmallPtrSet.cpp
  SmallVector.cpp
  SourceMgr.cpp
  SpecialCaseList.cpp
  Statistic.cpp
  StreamingMemoryObject.cpp
  StringExtras.cpp
  StringMap.cpp
  StringPool.cpp
  StringSaver.cpp
  StringRef.cpp
  SystemUtils.cpp
  TargetParser.cpp
  ThreadPool.cpp
  Timer.cpp
  ToolOutputFile.cpp
  Triple.cpp
  Twine.cpp
  Unicode.cpp
  YAMLParser.cpp
  YAMLTraits.cpp
  raw_os_ostream.cpp
  raw_ostream.cpp
  regcomp.c
  regerror.c
  regexec.c
  regfree.c
  regstrlcpy.c

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
  SearchForAddressOfSpecialSymbol.cpp
  Signals.cpp
  TargetRegistry.cpp
  ThreadLocal.cpp
  Threading.cpp
  TimeValue.cpp
  Valgrind.cpp
  Watchdog.cpp

  ADDITIONAL_HEADER_DIRS
  Unix
  Windows
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Support

  LINK_LIBS ${system_libs}
  )

set_property(TARGET LLVMSupport PROPERTY LLVM_SYSTEM_LIBS "${system_libs}")
