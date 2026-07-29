; REQUIRES: have_tf_aot
; REQUIRES: aarch64-registered-target
;
; Check that AArch64 virtual registers with copy hints do not get duplicate
; physical register hints added to AllocationOrder, which would otherwise lead
; to out-of-bounds indexing in MLEvictAdvisor.
;
; RUN: llc -mtriple=aarch64-linux-gnu -regalloc=greedy \
; RUN:   -regalloc-enable-advisor=release < %s -o /dev/null

%struct.Flags = type { i8, i32, i8, i8, i8, i8, i8, i32, i8 }

@asan_flags_dont_use_directly = external global %struct.Flags

define void @_Z15InitializeFlagsv(ptr %0, ptr %1) {
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @asan_flags_dont_use_directly, i64 4), align 4
  %3 = load i8, ptr getelementptr inbounds nuw (i8, ptr @asan_flags_dont_use_directly, i64 9), align 1
  store i8 %3, ptr getelementptr inbounds nuw (i8, ptr @asan_flags_dont_use_directly, i64 8), align 4
  %4 = load i8, ptr getelementptr inbounds nuw (i8, ptr @asan_flags_dont_use_directly, i64 12), align 4
  store i8 %4, ptr getelementptr inbounds nuw (i8, ptr @asan_flags_dont_use_directly, i64 11), align 1
  store i8 1, ptr getelementptr inbounds nuw (i8, ptr @asan_flags_dont_use_directly, i64 10), align 2
  %5 = load i8, ptr getelementptr inbounds nuw (i8, ptr @asan_flags_dont_use_directly, i64 20), align 4
  %6 = zext i8 %5 to i32
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @asan_flags_dont_use_directly, i64 16), align 4
  %7 = tail call ptr @_Znwmi(i64 16, i32 0)
  store ptr null, ptr %0, align 8
  store ptr null, ptr %7, align 8
  store ptr null, ptr @_Z26GetGlobalLowLevelAllocatorv, align 8
  %8 = tail call ptr @_Znwmi(i64 0, i32 0)
  store ptr @asan_flags_dont_use_directly, ptr %0, align 8
  %9 = tail call ptr @_Znwmi(i64 0, i32 0)
  store ptr getelementptr inbounds nuw (i8, ptr @asan_flags_dont_use_directly, i64 4), ptr %9, align 8
  %10 = load i32, ptr null, align 4
  %11 = tail call ptr @_Znwmi(i64 0, i32 %10)
  store ptr getelementptr inbounds nuw (i8, ptr @asan_flags_dont_use_directly, i64 8), ptr %0, align 8
  %12 = load i32, ptr null, align 4
  %13 = tail call ptr @_Znwmi(i64 0, i32 %12)
  store ptr getelementptr inbounds nuw (i8, ptr @asan_flags_dont_use_directly, i64 9), ptr @_Znwmi, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @asan_flags_dont_use_directly, i64 10), ptr %1, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @asan_flags_dont_use_directly, i64 11), ptr %0, align 8
  %14 = tail call ptr @_Znwmi(i64 0, i32 0)
  store ptr getelementptr inbounds nuw (i8, ptr @asan_flags_dont_use_directly, i64 12), ptr %0, align 8
  %15 = tail call ptr @_Znwmi(i64 0, i32 0)
  store ptr getelementptr inbounds nuw (i8, ptr @asan_flags_dont_use_directly, i64 16), ptr %1, align 8
  store ptr getelementptr inbounds nuw (i8, ptr @asan_flags_dont_use_directly, i64 20), ptr %0, align 8
  ret void
}

declare i32 @_Z26GetGlobalLowLevelAllocatorv()

declare ptr @_Znwmi(i64, i32)

; uselistorder directives
uselistorder ptr @_Znwmi, { 7, 6, 5, 4, 3, 2, 1, 0 }
