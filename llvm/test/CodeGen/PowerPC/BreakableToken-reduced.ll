; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 -enable-shrink-wrap=true %s -o - | FileCheck %s
;
; Test the use of a non-R0 register to save/restore the LR in function 
; prologue/epilogue.
; This problem can occur as a result of shrink wrapping, where the function
; prologue and epilogue are moved from the beginning/ending of the function. If
; register R0 is used before the prologue/epilogue blocks, then it cannot be
; used to save/restore the LR.
;
; TODO: Convert this to an MIR test once the infrastructure can support it.
;       To convert this to an MIR pass, generate MIR after register allocation
;       but before shrink wrapping and verify that has been used in the body of
;       the function. This can be done with something like: 
;         llc -stop-after stack-slot-coloring BreakableToken-reduced.ll > BreakableToken-reduced.mir
;
;       The resulting MIR file can then be used as input to llc, and only run
;       shrink wrapping and Prologue/Epilogue insertion on it. For example:
;         llc -start-after stack-slot-coloring -stop-after prologepilog BreakableToken-reduced.mir
;       
;       Verify in the resulting code that R0 is not used in the prologue/epilogue.
;
;       This currently cannot be done because the PrologEpilogInserter pass has
;       a dependency on the TargetPassConfig and StackProtector classes, which
;       are currently not serialized when generating the MIR.
;

; ModuleID = 'BreakableToken.cpp'
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

%"class.clang::format::BreakableStringLiteral" = type { %"class.clang::format::BreakableSingleLineToken" }
%"class.clang::format::BreakableSingleLineToken" = type { %"class.clang::format::BreakableToken", i32, %"class.llvm::StringRef", %"class.llvm::StringRef", %"class.llvm::StringRef" }
%"class.clang::format::BreakableToken" = type { i32 (...)**, %"struct.clang::format::FormatToken"*, i32, i8, i32, %"struct.clang::format::FormatStyle"* }
%"class.llvm::StringRef" = type { i8*, i64 }
%"struct.clang::format::FormatToken" = type <{ %"class.clang::Token", i32, i8, [3 x i8], %"class.clang::SourceRange", i32, i32, i32, i8, i8, i8, i8, %"class.llvm::StringRef", i8, [3 x i8], i32, i32, i32, i8, i8, [2 x i8], i32, i32, i16, [2 x i8], %"class.std::unique_ptr", i32, i32, i32, i32, i32, i32, i32, i32, %"class.llvm::SmallVector", i32, i8, i8, [2 x i8], i32, i8, i8, [2 x i8], %"struct.clang::format::FormatToken"*, %"struct.clang::format::FormatToken"*, %"struct.clang::format::FormatToken"*, %"class.llvm::SmallVector.6", i32, i8, [3 x i8] }>
%"class.clang::Token" = type <{ i32, i32, i8*, i16, i16, [4 x i8] }>
%"class.clang::SourceRange" = type { %"class.clang::SourceLocation", %"class.clang::SourceLocation" }
%"class.clang::SourceLocation" = type { i32 }
%"class.std::unique_ptr" = type { %"class.std::tuple" }
%"class.std::tuple" = type { %"struct.std::_Tuple_impl" }
%"struct.std::_Tuple_impl" = type { %"struct.std::_Head_base.2" }
%"struct.std::_Head_base.2" = type { %"class.clang::format::TokenRole"* }
%"class.clang::format::TokenRole" = type { i32 (...)**, %"struct.clang::format::FormatStyle"* }
%"class.llvm::SmallVector" = type { %"class.llvm::SmallVectorImpl.base", %"struct.llvm::SmallVectorStorage" }
%"class.llvm::SmallVectorImpl.base" = type { %"class.llvm::SmallVectorTemplateBase.base" }
%"class.llvm::SmallVectorTemplateBase.base" = type { %"class.llvm::SmallVectorTemplateCommon.base" }
%"class.llvm::SmallVectorTemplateCommon.base" = type <{ %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion" }>
%"class.llvm::SmallVectorBase" = type { i8*, i8*, i8* }
%"struct.llvm::AlignedCharArrayUnion" = type { %"struct.llvm::AlignedCharArray" }
%"struct.llvm::AlignedCharArray" = type { [4 x i8] }
%"struct.llvm::SmallVectorStorage" = type { [3 x %"struct.llvm::AlignedCharArrayUnion"] }
%"class.llvm::SmallVector.6" = type <{ %"class.llvm::SmallVectorImpl.7", %"struct.llvm::SmallVectorStorage.12", [7 x i8] }>
%"class.llvm::SmallVectorImpl.7" = type { %"class.llvm::SmallVectorTemplateBase.8" }
%"class.llvm::SmallVectorTemplateBase.8" = type { %"class.llvm::SmallVectorTemplateCommon.9" }
%"class.llvm::SmallVectorTemplateCommon.9" = type { %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion.10" }
%"struct.llvm::AlignedCharArrayUnion.10" = type { %"struct.llvm::AlignedCharArray.11" }
%"struct.llvm::AlignedCharArray.11" = type { [8 x i8] }
%"struct.llvm::SmallVectorStorage.12" = type { i8 }
%"struct.clang::format::FormatStyle" = type { i32, i8, i8, i8, i8, i8, i8, i8, i8, i32, i8, i8, i32, i8, i8, i8, i8, i32, i32, i8, i8, i32, %"class.std::basic_string", i8, i32, i32, i8, i8, i8, i8, %"class.std::vector", i8, i32, i8, i8, i32, %"class.std::basic_string", %"class.std::basic_string", i32, i32, i32, i8, i8, i32, i32, i32, i32, i32, i32, i32, i8, i8, i32, i8, i32, i8, i8, i8, i8, i8, i32, i32, i32 }
%"class.std::vector" = type { %"struct.std::_Vector_base" }
%"struct.std::_Vector_base" = type { %"struct.std::_Vector_base<std::basic_string<char>, std::allocator<std::basic_string<char> > >::_Vector_impl" }
%"struct.std::_Vector_base<std::basic_string<char>, std::allocator<std::basic_string<char> > >::_Vector_impl" = type { %"class.std::basic_string"*, %"class.std::basic_string"*, %"class.std::basic_string"* }
%"class.std::basic_string" = type { %"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" }
%"struct.std::basic_string<char, std::char_traits<char>, std::allocator<char> >::_Alloc_hider" = type { i8* }
%"struct.llvm::AlignedCharArray.52" = type { [16 x i8] }
%"class.clang::format::WhitespaceManager" = type <{ %"class.llvm::SmallVector.13", %"class.clang::SourceManager"*, %"class.std::set", %"struct.clang::format::FormatStyle"*, i8, [7 x i8] }>
%"class.llvm::SmallVector.13" = type { %"class.llvm::SmallVectorImpl.14", %"struct.llvm::SmallVectorStorage.19" }
%"class.llvm::SmallVectorImpl.14" = type { %"class.llvm::SmallVectorTemplateBase.15" }
%"class.llvm::SmallVectorTemplateBase.15" = type { %"class.llvm::SmallVectorTemplateCommon.16" }
%"class.llvm::SmallVectorTemplateCommon.16" = type { %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion.17" }
%"struct.llvm::AlignedCharArrayUnion.17" = type { %"struct.llvm::AlignedCharArray.18" }
%"struct.llvm::AlignedCharArray.18" = type { [88 x i8] }
%"struct.llvm::SmallVectorStorage.19" = type { [15 x %"struct.llvm::AlignedCharArrayUnion.17"] }
%"class.clang::SourceManager" = type { %"class.llvm::RefCountedBase", %"class.clang::DiagnosticsEngine"*, %"class.clang::FileManager"*, %"class.llvm::BumpPtrAllocatorImpl", %"class.llvm::DenseMap.65", i8, i8, %"class.std::unique_ptr.78", %"class.std::vector.94", %"class.llvm::SmallVector.99", %"class.llvm::SmallVector.99", i32, i32, %"class.std::vector.107", %"class.clang::ExternalSLocEntrySource"*, %"class.clang::FileID", %"class.clang::LineTableInfo"*, %"class.clang::FileID", %"class.clang::SrcMgr::ContentCache"*, i32, i32, %"class.clang::FileID", %"class.clang::FileID", i32, i32, %"class.llvm::DenseMap.111", %"class.llvm::DenseMap.115", %"class.clang::InBeforeInTUCacheEntry", %"class.std::unique_ptr.119", %"class.std::unique_ptr.127", %"class.llvm::DenseMap.135", %"class.llvm::SmallVector.139" }
%"class.llvm::RefCountedBase" = type { i32 }
%"class.clang::DiagnosticsEngine" = type opaque
%"class.clang::FileManager" = type { %"class.llvm::RefCountedBase.20", %"class.llvm::IntrusiveRefCntPtr", %"class.clang::FileSystemOptions", %"class.std::map", %"class.std::map.24", %"class.llvm::SmallVector.29", %"class.llvm::SmallVector.35", %"class.llvm::StringMap", %"class.llvm::StringMap.56", %"class.llvm::DenseMap", %"class.llvm::BumpPtrAllocatorImpl", i32, i32, i32, i32, i32, %"class.std::unique_ptr.57" }
%"class.llvm::RefCountedBase.20" = type { i32 }
%"class.llvm::IntrusiveRefCntPtr" = type { %"class.clang::vfs::FileSystem"* }
%"class.clang::vfs::FileSystem" = type <{ i32 (...)**, %"class.llvm::ThreadSafeRefCountedBase", [4 x i8] }>
%"class.llvm::ThreadSafeRefCountedBase" = type { %"struct.std::atomic" }
%"struct.std::atomic" = type { %"struct.std::__atomic_base" }
%"struct.std::__atomic_base" = type { i32 }
%"class.clang::FileSystemOptions" = type { %"class.std::basic_string" }
%"class.std::map" = type { %"class.std::_Rb_tree" }
%"class.std::_Rb_tree" = type { %"struct.std::_Rb_tree<llvm::sys::fs::UniqueID, std::pair<const llvm::sys::fs::UniqueID, clang::DirectoryEntry>, std::_Select1st<std::pair<const llvm::sys::fs::UniqueID, clang::DirectoryEntry> >, std::less<llvm::sys::fs::UniqueID>, std::allocator<std::pair<const llvm::sys::fs::UniqueID, clang::DirectoryEntry> > >::_Rb_tree_impl" }
%"struct.std::_Rb_tree<llvm::sys::fs::UniqueID, std::pair<const llvm::sys::fs::UniqueID, clang::DirectoryEntry>, std::_Select1st<std::pair<const llvm::sys::fs::UniqueID, clang::DirectoryEntry> >, std::less<llvm::sys::fs::UniqueID>, std::allocator<std::pair<const llvm::sys::fs::UniqueID, clang::DirectoryEntry> > >::_Rb_tree_impl" = type { %"struct.std::less", %"struct.std::_Rb_tree_node_base", i64 }
%"struct.std::less" = type { i8 }
%"struct.std::_Rb_tree_node_base" = type { i32, %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"*, %"struct.std::_Rb_tree_node_base"* }
%"class.std::map.24" = type { %"class.std::_Rb_tree.25" }
%"class.std::_Rb_tree.25" = type { %"struct.std::_Rb_tree<llvm::sys::fs::UniqueID, std::pair<const llvm::sys::fs::UniqueID, clang::FileEntry>, std::_Select1st<std::pair<const llvm::sys::fs::UniqueID, clang::FileEntry> >, std::less<llvm::sys::fs::UniqueID>, std::allocator<std::pair<const llvm::sys::fs::UniqueID, clang::FileEntry> > >::_Rb_tree_impl" }
%"struct.std::_Rb_tree<llvm::sys::fs::UniqueID, std::pair<const llvm::sys::fs::UniqueID, clang::FileEntry>, std::_Select1st<std::pair<const llvm::sys::fs::UniqueID, clang::FileEntry> >, std::less<llvm::sys::fs::UniqueID>, std::allocator<std::pair<const llvm::sys::fs::UniqueID, clang::FileEntry> > >::_Rb_tree_impl" = type { %"struct.std::less", %"struct.std::_Rb_tree_node_base", i64 }
%"class.llvm::SmallVector.29" = type { %"class.llvm::SmallVectorImpl.30", %"struct.llvm::SmallVectorStorage.34" }
%"class.llvm::SmallVectorImpl.30" = type { %"class.llvm::SmallVectorTemplateBase.31" }
%"class.llvm::SmallVectorTemplateBase.31" = type { %"class.llvm::SmallVectorTemplateCommon.32" }
%"class.llvm::SmallVectorTemplateCommon.32" = type { %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion.33" }
%"struct.llvm::AlignedCharArrayUnion.33" = type { %"struct.llvm::AlignedCharArray.11" }
%"struct.llvm::SmallVectorStorage.34" = type { [3 x %"struct.llvm::AlignedCharArrayUnion.33"] }
%"class.llvm::SmallVector.35" = type { %"class.llvm::SmallVectorImpl.36", %"struct.llvm::SmallVectorStorage.40" }
%"class.llvm::SmallVectorImpl.36" = type { %"class.llvm::SmallVectorTemplateBase.37" }
%"class.llvm::SmallVectorTemplateBase.37" = type { %"class.llvm::SmallVectorTemplateCommon.38" }
%"class.llvm::SmallVectorTemplateCommon.38" = type { %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion.39" }
%"struct.llvm::AlignedCharArrayUnion.39" = type { %"struct.llvm::AlignedCharArray.11" }
%"struct.llvm::SmallVectorStorage.40" = type { [3 x %"struct.llvm::AlignedCharArrayUnion.39"] }
%"class.llvm::StringMap" = type { %"class.llvm::StringMapImpl", %"class.llvm::BumpPtrAllocatorImpl" }
%"class.llvm::StringMapImpl" = type { %"class.llvm::StringMapEntryBase"**, i32, i32, i32, i32 }
%"class.llvm::StringMapEntryBase" = type { i32 }
%"class.llvm::StringMap.56" = type { %"class.llvm::StringMapImpl", %"class.llvm::BumpPtrAllocatorImpl" }
%"class.llvm::DenseMap" = type <{ %"struct.llvm::detail::DenseMapPair"*, i32, i32, i32, [4 x i8] }>
%"struct.llvm::detail::DenseMapPair" = type opaque
%"class.std::unique_ptr.57" = type { %"class.std::tuple.58" }
%"class.std::tuple.58" = type { %"struct.std::_Tuple_impl.59" }
%"struct.std::_Tuple_impl.59" = type { %"struct.std::_Head_base.64" }
%"struct.std::_Head_base.64" = type { %"class.clang::FileSystemStatCache"* }
%"class.clang::FileSystemStatCache" = type opaque
%"class.llvm::BumpPtrAllocatorImpl" = type <{ i8*, i8*, %"class.llvm::SmallVector.41", %"class.llvm::SmallVector.47", i64, %"class.llvm::MallocAllocator", [7 x i8] }>
%"class.llvm::SmallVector.41" = type { %"class.llvm::SmallVectorImpl.42", %"struct.llvm::SmallVectorStorage.46" }
%"class.llvm::SmallVectorImpl.42" = type { %"class.llvm::SmallVectorTemplateBase.43" }
%"class.llvm::SmallVectorTemplateBase.43" = type { %"class.llvm::SmallVectorTemplateCommon.44" }
%"class.llvm::SmallVectorTemplateCommon.44" = type { %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion.45" }
%"struct.llvm::AlignedCharArrayUnion.45" = type { %"struct.llvm::AlignedCharArray.11" }
%"struct.llvm::SmallVectorStorage.46" = type { [3 x %"struct.llvm::AlignedCharArrayUnion.45"] }
%"class.llvm::SmallVector.47" = type <{ %"class.llvm::SmallVectorImpl.48", %"struct.llvm::SmallVectorStorage.53", [7 x i8] }>
%"class.llvm::SmallVectorImpl.48" = type { %"class.llvm::SmallVectorTemplateBase.49" }
%"class.llvm::SmallVectorTemplateBase.49" = type { %"class.llvm::SmallVectorTemplateCommon.50" }
%"class.llvm::SmallVectorTemplateCommon.50" = type { %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion.51" }
%"struct.llvm::AlignedCharArrayUnion.51" = type { %"struct.llvm::AlignedCharArray.52" }
%"struct.llvm::SmallVectorStorage.53" = type { i8 }
%"class.llvm::MallocAllocator" = type { i8 }
%"class.llvm::DenseMap.65" = type <{ %"struct.llvm::detail::DenseMapPair.67"*, i32, i32, i32, [4 x i8] }>
%"struct.llvm::detail::DenseMapPair.67" = type { %"struct.std::pair.68" }
%"struct.std::pair.68" = type { %"class.clang::FileEntry"*, %"class.clang::SrcMgr::ContentCache"* }
%"class.clang::FileEntry" = type { i8*, i64, i64, %"class.clang::DirectoryEntry"*, i32, %"class.llvm::sys::fs::UniqueID", i8, i8, i8, %"class.std::unique_ptr.69" }
%"class.clang::DirectoryEntry" = type { i8* }
%"class.llvm::sys::fs::UniqueID" = type { i64, i64 }
%"class.std::unique_ptr.69" = type { %"class.std::tuple.70" }
%"class.std::tuple.70" = type { %"struct.std::_Tuple_impl.71" }
%"struct.std::_Tuple_impl.71" = type { %"struct.std::_Head_base.76" }
%"struct.std::_Head_base.76" = type { %"class.clang::vfs::File"* }
%"class.clang::vfs::File" = type { i32 (...)** }
%"class.std::unique_ptr.78" = type { %"class.std::tuple.79" }
%"class.std::tuple.79" = type { %"struct.std::_Tuple_impl.80" }
%"struct.std::_Tuple_impl.80" = type { %"struct.std::_Head_base.85" }
%"struct.std::_Head_base.85" = type { %"struct.clang::SourceManager::OverriddenFilesInfoTy"* }
%"struct.clang::SourceManager::OverriddenFilesInfoTy" = type { %"class.llvm::DenseMap.86", %"class.llvm::DenseSet" }
%"class.llvm::DenseMap.86" = type <{ %"struct.llvm::detail::DenseMapPair.88"*, i32, i32, i32, [4 x i8] }>
%"struct.llvm::detail::DenseMapPair.88" = type { %"struct.std::pair.89" }
%"struct.std::pair.89" = type { %"class.clang::FileEntry"*, %"class.clang::FileEntry"* }
%"class.llvm::DenseSet" = type { %"class.llvm::DenseMap.91" }
%"class.llvm::DenseMap.91" = type <{ %"class.llvm::detail::DenseSetPair"*, i32, i32, i32, [4 x i8] }>
%"class.llvm::detail::DenseSetPair" = type { %"class.clang::FileEntry"* }
%"class.std::vector.94" = type { %"struct.std::_Vector_base.95" }
%"struct.std::_Vector_base.95" = type { %"struct.std::_Vector_base<clang::SrcMgr::ContentCache *, std::allocator<clang::SrcMgr::ContentCache *> >::_Vector_impl" }
%"struct.std::_Vector_base<clang::SrcMgr::ContentCache *, std::allocator<clang::SrcMgr::ContentCache *> >::_Vector_impl" = type { %"class.clang::SrcMgr::ContentCache"**, %"class.clang::SrcMgr::ContentCache"**, %"class.clang::SrcMgr::ContentCache"** }
%"class.llvm::SmallVector.99" = type <{ %"class.llvm::SmallVectorImpl.100", %"struct.llvm::SmallVectorStorage.105", [7 x i8] }>
%"class.llvm::SmallVectorImpl.100" = type { %"class.llvm::SmallVectorTemplateBase.101" }
%"class.llvm::SmallVectorTemplateBase.101" = type { %"class.llvm::SmallVectorTemplateCommon.102" }
%"class.llvm::SmallVectorTemplateCommon.102" = type { %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion.103" }
%"struct.llvm::AlignedCharArrayUnion.103" = type { %"struct.llvm::AlignedCharArray.104" }
%"struct.llvm::AlignedCharArray.104" = type { [24 x i8] }
%"struct.llvm::SmallVectorStorage.105" = type { i8 }
%"class.std::vector.107" = type { %"struct.std::_Bvector_base" }
%"struct.std::_Bvector_base" = type { %"struct.std::_Bvector_base<std::allocator<bool> >::_Bvector_impl" }
%"struct.std::_Bvector_base<std::allocator<bool> >::_Bvector_impl" = type { %"struct.std::_Bit_iterator", %"struct.std::_Bit_iterator", i64* }
%"struct.std::_Bit_iterator" = type { %"struct.std::_Bit_iterator_base.base", [4 x i8] }
%"struct.std::_Bit_iterator_base.base" = type <{ i64*, i32 }>
%"class.clang::ExternalSLocEntrySource" = type { i32 (...)** }
%"class.clang::LineTableInfo" = type opaque
%"class.clang::SrcMgr::ContentCache" = type <{ %"class.llvm::PointerIntPair", %"class.clang::FileEntry"*, %"class.clang::FileEntry"*, i32*, [5 x i8], [3 x i8] }>
%"class.llvm::PointerIntPair" = type { i64 }
%"class.clang::FileID" = type { i32 }
%"class.llvm::DenseMap.111" = type <{ %"struct.llvm::detail::DenseMapPair.113"*, i32, i32, i32, [4 x i8] }>
%"struct.llvm::detail::DenseMapPair.113" = type opaque
%"class.llvm::DenseMap.115" = type <{ %"struct.llvm::detail::DenseMapPair.117"*, i32, i32, i32, [4 x i8] }>
%"struct.llvm::detail::DenseMapPair.117" = type opaque
%"class.clang::InBeforeInTUCacheEntry" = type { %"class.clang::FileID", %"class.clang::FileID", i8, %"class.clang::FileID", i32, i32 }
%"class.std::unique_ptr.119" = type { %"class.std::tuple.120" }
%"class.std::tuple.120" = type { %"struct.std::_Tuple_impl.121" }
%"struct.std::_Tuple_impl.121" = type { %"struct.std::_Head_base.126" }
%"struct.std::_Head_base.126" = type { %"class.llvm::MemoryBuffer"* }
%"class.llvm::MemoryBuffer" = type { i32 (...)**, i8*, i8* }
%"class.std::unique_ptr.127" = type { %"class.std::tuple.128" }
%"class.std::tuple.128" = type { %"struct.std::_Tuple_impl.129" }
%"struct.std::_Tuple_impl.129" = type { %"struct.std::_Head_base.134" }
%"struct.std::_Head_base.134" = type { %"class.clang::SrcMgr::ContentCache"* }
%"class.llvm::DenseMap.135" = type <{ %"struct.llvm::detail::DenseMapPair.137"*, i32, i32, i32, [4 x i8] }>
%"struct.llvm::detail::DenseMapPair.137" = type opaque
%"class.llvm::SmallVector.139" = type { %"class.llvm::SmallVectorImpl.140", %"struct.llvm::SmallVectorStorage.144" }
%"class.llvm::SmallVectorImpl.140" = type { %"class.llvm::SmallVectorTemplateBase.141" }
%"class.llvm::SmallVectorTemplateBase.141" = type { %"class.llvm::SmallVectorTemplateCommon.142" }
%"class.llvm::SmallVectorTemplateCommon.142" = type { %"class.llvm::SmallVectorBase", %"struct.llvm::AlignedCharArrayUnion.143" }
%"struct.llvm::AlignedCharArrayUnion.143" = type { %"struct.llvm::AlignedCharArray.104" }
%"struct.llvm::SmallVectorStorage.144" = type { [1 x %"struct.llvm::AlignedCharArrayUnion.143"] }
%"class.std::set" = type { %"class.std::_Rb_tree.145" }
%"class.std::_Rb_tree.145" = type { %"struct.std::_Rb_tree<clang::tooling::Replacement, clang::tooling::Replacement, std::_Identity<clang::tooling::Replacement>, std::less<clang::tooling::Replacement>, std::allocator<clang::tooling::Replacement> >::_Rb_tree_impl" }
%"struct.std::_Rb_tree<clang::tooling::Replacement, clang::tooling::Replacement, std::_Identity<clang::tooling::Replacement>, std::less<clang::tooling::Replacement>, std::allocator<clang::tooling::Replacement> >::_Rb_tree_impl" = type { %"struct.std::less.149", %"struct.std::_Rb_tree_node_base", i64 }
%"struct.std::less.149" = type { i8 }


; Function Attrs: nounwind
; CHECK-LABEL: @_ZN5clang6format22BreakableStringLiteral11insertBreakEjjSt4pairImjERNS0_17WhitespaceManagerE

; Load a value into R0 before saving the LR
; CHECK: lwz 0, {{[0-9]+([0-9]+)}}

; Ensure the LR is saved using a different register - edit:D63152 prevents stack pop befor loads and stores
; CHECK-NOT: mflr {{[1-9]+}}

; Ensure the LR is restored using a different register
; CHECK: mtlr {{[0-9]+}}
; CHECK: blr
define void @_ZN5clang6format22BreakableStringLiteral11insertBreakEjjSt4pairImjERNS0_17WhitespaceManagerE(%"class.clang::format::BreakableStringLiteral"* nocapture readonly %this, i32 zeroext %LineIndex, i32 zeroext %TailOffset, [2 x i64] %Split.coerce, %"class.clang::format::WhitespaceManager"* dereferenceable(1504) %Whitespaces) unnamed_addr #1 align 2 {
entry:
  %Split.coerce.fca.0.extract = extractvalue [2 x i64] %Split.coerce, 0
  %Split.coerce.fca.1.extract = extractvalue [2 x i64] %Split.coerce, 1
  %StartColumn = getelementptr inbounds %"class.clang::format::BreakableStringLiteral", %"class.clang::format::BreakableStringLiteral"* %this, i64 0, i32 0, i32 1
  %0 = load i32, i32* %StartColumn, align 8, !tbaa !2
  %Prefix = getelementptr inbounds %"class.clang::format::BreakableStringLiteral", %"class.clang::format::BreakableStringLiteral"* %this, i64 0, i32 0, i32 2
  %Length.i.19 = getelementptr inbounds %"class.clang::format::BreakableStringLiteral", %"class.clang::format::BreakableStringLiteral"* %this, i64 0, i32 0, i32 2, i32 1
  %1 = load i64, i64* %Length.i.19, align 8, !tbaa !10
  %cmp.i = icmp eq i64 %1, 0
  br i1 %cmp.i, label %entry._ZNK4llvm9StringRef10startswithES0_.exit_crit_edge, label %if.end.i.i

entry._ZNK4llvm9StringRef10startswithES0_.exit_crit_edge: ; preds = %entry
  %agg.tmp7.sroa.0.0..sroa_cast.phi.trans.insert = bitcast %"class.llvm::StringRef"* %Prefix to i64*
  %agg.tmp7.sroa.0.0.copyload.pre = load i64, i64* %agg.tmp7.sroa.0.0..sroa_cast.phi.trans.insert, align 8
  br label %_ZNK4llvm9StringRef10startswithES0_.exit

if.end.i.i:                                       ; preds = %entry
  %Data.i.20 = getelementptr inbounds %"class.llvm::StringRef", %"class.llvm::StringRef"* %Prefix, i64 0, i32 0
  %2 = load i8*, i8** %Data.i.20, align 8, !tbaa !12
  %lhsc = load i8, i8* %2, align 1
  %phitmp.i = icmp eq i8 %lhsc, 64
  %3 = ptrtoint i8* %2 to i64
  br label %_ZNK4llvm9StringRef10startswithES0_.exit

_ZNK4llvm9StringRef10startswithES0_.exit:         ; preds = %entry._ZNK4llvm9StringRef10startswithES0_.exit_crit_edge, %if.end.i.i
  %agg.tmp7.sroa.0.0.copyload = phi i64 [ %agg.tmp7.sroa.0.0.copyload.pre, %entry._ZNK4llvm9StringRef10startswithES0_.exit_crit_edge ], [ %3, %if.end.i.i ]
  %4 = phi i1 [ false, %entry._ZNK4llvm9StringRef10startswithES0_.exit_crit_edge ], [ %phitmp.i, %if.end.i.i ]
  %dec = sext i1 %4 to i32
  %dec. = add i32 %dec, %0
  %Tok = getelementptr inbounds %"class.clang::format::BreakableStringLiteral", %"class.clang::format::BreakableStringLiteral"* %this, i64 0, i32 0, i32 0, i32 1
  %ref = load %"struct.clang::format::FormatToken"*, %"struct.clang::format::FormatToken"** %Tok, align 8, !tbaa !13
  %conv = zext i32 %TailOffset to i64
  %add = add i64 %Split.coerce.fca.0.extract, %conv
  %add4 = add i64 %add, %1
  %conv5 = trunc i64 %add4 to i32
  %Split.sroa.2.8.extract.trunc = trunc i64 %Split.coerce.fca.1.extract to i32
  %agg.tmp6.sroa.0.0..sroa_idx13 = getelementptr inbounds %"class.clang::format::BreakableStringLiteral", %"class.clang::format::BreakableStringLiteral"* %this, i64 0, i32 0, i32 3
  %agg.tmp6.sroa.0.0..sroa_cast = bitcast %"class.llvm::StringRef"* %agg.tmp6.sroa.0.0..sroa_idx13 to i64*
  %agg.tmp6.sroa.0.0.copyload = load i64, i64* %agg.tmp6.sroa.0.0..sroa_cast, align 8
  %agg.tmp6.sroa.2.0..sroa_idx14 = getelementptr inbounds %"class.clang::format::BreakableStringLiteral", %"class.clang::format::BreakableStringLiteral"* %this, i64 0, i32 0, i32 3, i32 1
  %agg.tmp6.sroa.2.0.copyload = load i64, i64* %agg.tmp6.sroa.2.0..sroa_idx14, align 8
  %InPPDirective = getelementptr inbounds %"class.clang::format::BreakableStringLiteral", %"class.clang::format::BreakableStringLiteral"* %this, i64 0, i32 0, i32 0, i32 3
  %5 = load i8, i8* %InPPDirective, align 4, !tbaa !34, !range !39
  %tobool = icmp ne i8 %5, 0
  %IndentLevel = getelementptr inbounds %"class.clang::format::BreakableStringLiteral", %"class.clang::format::BreakableStringLiteral"* %this, i64 0, i32 0, i32 0, i32 2
  %6 = load i32, i32* %IndentLevel, align 8, !tbaa !33
  %.fca.0.insert11 = insertvalue [2 x i64] undef, i64 %agg.tmp6.sroa.0.0.copyload, 0
  %.fca.1.insert12 = insertvalue [2 x i64] %.fca.0.insert11, i64 %agg.tmp6.sroa.2.0.copyload, 1
  %.fca.0.insert = insertvalue [2 x i64] undef, i64 %agg.tmp7.sroa.0.0.copyload, 0
  %.fca.1.insert = insertvalue [2 x i64] %.fca.0.insert, i64 %1, 1
  tail call void @_ZN5clang6format17WhitespaceManager24replaceWhitespaceInTokenERKNS0_11FormatTokenEjjN4llvm9StringRefES6_bjji(%"class.clang::format::WhitespaceManager"* nonnull %Whitespaces, %"struct.clang::format::FormatToken"* dereferenceable(272) %ref, i32 zeroext %conv5, i32 zeroext %Split.sroa.2.8.extract.trunc, [2 x i64] %.fca.1.insert12, [2 x i64] %.fca.1.insert, i1 zeroext %tobool, i32 zeroext 1, i32 zeroext %6, i32 signext %dec.) #9
  ret void
}

; Function Attrs: nounwind argmemonly
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #2

declare void @_ZN5clang6format17WhitespaceManager24replaceWhitespaceInTokenERKNS0_11FormatTokenEjjN4llvm9StringRefES6_bjji(%"class.clang::format::WhitespaceManager"*, %"struct.clang::format::FormatToken"* dereferenceable(272), i32 zeroext, i32 zeroext, [2 x i64], [2 x i64], i1 zeroext, i32 zeroext, i32 zeroext, i32 signext) #3

; Function Attrs: nounwind argmemonly
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #2

attributes #9 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 3.8.0 (trunk 248714) (llvm/trunk 248719)"}
!2 = !{!3, !4, i64 40}
!3 = !{!"_ZTSN5clang6format24BreakableSingleLineTokenE", !4, i64 40, !7, i64 48, !7, i64 64, !7, i64 80}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!"_ZTSN4llvm9StringRefE", !8, i64 0, !9, i64 8}
!8 = !{!"any pointer", !5, i64 0}
!9 = !{!"long", !5, i64 0}
!10 = !{!7, !9, i64 8}
!11 = !{!9, !9, i64 0}
!12 = !{!7, !8, i64 0}
!13 = !{!5, !5, i64 0}
!14 = !{!15, !4, i64 200}
!15 = !{!"_ZTSN5clang6format11FormatStyleE", !4, i64 0, !16, i64 4, !16, i64 5, !16, i64 6, !16, i64 7, !16, i64 8, !16, i64 9, !16, i64 10, !16, i64 11, !17, i64 12, !16, i64 16, !16, i64 17, !18, i64 20, !16, i64 24, !16, i64 25, !16, i64 26, !16, i64 27, !19, i64 28, !20, i64 32, !16, i64 36, !16, i64 37, !4, i64 40, !21, i64 48, !16, i64 56, !4, i64 60, !4, i64 64, !16, i64 68, !16, i64 69, !16, i64 70, !16, i64 71, !23, i64 72, !16, i64 96, !4, i64 100, !16, i64 104, !16, i64 105, !24, i64 108, !21, i64 112, !21, i64 120, !4, i64 128, !25, i64 132, !4, i64 136, !16, i64 140, !16, i64 141, !4, i64 144, !4, i64 148, !4, i64 152, !4, i64 156, !4, i64 160, !4, i64 164, !26, i64 168, !16, i64 172, !16, i64 173, !27, i64 176, !16, i64 180, !4, i64 184, !16, i64 188, !16, i64 189, !16, i64 190, !16, i64 191, !16, i64 192, !28, i64 196, !4, i64 200, !29, i64 204}
!16 = !{!"bool", !5, i64 0}
!17 = !{!"_ZTSN5clang6format11FormatStyle18ShortFunctionStyleE", !5, i64 0}
!18 = !{!"_ZTSN5clang6format11FormatStyle33DefinitionReturnTypeBreakingStyleE", !5, i64 0}
!19 = !{!"_ZTSN5clang6format11FormatStyle19BinaryOperatorStyleE", !5, i64 0}
!20 = !{!"_ZTSN5clang6format11FormatStyle18BraceBreakingStyleE", !5, i64 0}
!21 = !{!"_ZTSSs", !22, i64 0}
!22 = !{!"_ZTSNSs12_Alloc_hiderE", !8, i64 0}
!23 = !{!"_ZTSSt6vectorISsSaISsEE"}
!24 = !{!"_ZTSN5clang6format11FormatStyle12LanguageKindE", !5, i64 0}
!25 = !{!"_ZTSN5clang6format11FormatStyle24NamespaceIndentationKindE", !5, i64 0}
!26 = !{!"_ZTSN5clang6format11FormatStyle21PointerAlignmentStyleE", !5, i64 0}
!27 = !{!"_ZTSN5clang6format11FormatStyle24SpaceBeforeParensOptionsE", !5, i64 0}
!28 = !{!"_ZTSN5clang6format11FormatStyle16LanguageStandardE", !5, i64 0}
!29 = !{!"_ZTSN5clang6format11FormatStyle11UseTabStyleE", !5, i64 0}
!30 = !{!31, !32, i64 24}
!31 = !{!"_ZTSN5clang6format14BreakableTokenE", !5, i64 8, !4, i64 16, !16, i64 20, !32, i64 24, !5, i64 32}
!32 = !{!"_ZTSN5clang6format8encoding8EncodingE", !5, i64 0}
!33 = !{!31, !4, i64 16}
!34 = !{!31, !16, i64 20}
!35 = !{!36, !36, i64 0}
!36 = !{!"vtable pointer", !6, i64 0}
!37 = !{!38, !38, i64 0}
!38 = !{!"short", !5, i64 0}
!39 = !{i8 0, i8 2}
!40 = !{i64 0, i64 8, !41, i64 8, i64 8, !11}
!41 = !{!8, !8, i64 0}
!42 = !{!43, !8, i64 16}
!43 = !{!"_ZTSN4llvm15SmallVectorBaseE", !8, i64 0, !8, i64 8, !8, i64 16}
!44 = !{!43, !8, i64 8}
!45 = !{!43, !8, i64 0}
!46 = !{!4, !4, i64 0}
!47 = !{!48, !16, i64 500}
!48 = !{!"_ZTSN5clang6format21BreakableBlockCommentE", !49, i64 40, !51, i64 320, !53, i64 408, !4, i64 496, !16, i64 500, !7, i64 504}
!49 = !{!"_ZTSN4llvm11SmallVectorINS_9StringRefELj16EEE", !50, i64 40}
!50 = !{!"_ZTSN4llvm18SmallVectorStorageINS_9StringRefELj16EEE", !5, i64 0}
!51 = !{!"_ZTSN4llvm11SmallVectorIjLj16EEE", !52, i64 28}
!52 = !{!"_ZTSN4llvm18SmallVectorStorageIjLj16EEE", !5, i64 0}
!53 = !{!"_ZTSN4llvm11SmallVectorIiLj16EEE", !54, i64 28}
!54 = !{!"_ZTSN4llvm18SmallVectorStorageIiLj16EEE", !5, i64 0}
!55 = !{!48, !4, i64 496}
