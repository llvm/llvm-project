; RUN: llc -mtriple=x86_64-pc-win32-macho -relocation-model=static -O0 < %s | FileCheck %s

; Ensure that we don't generate a movl and not a lea for a static relocation
; when compiling for 64 bit.

%struct.MatchInfo = type [64 x i64]

@NO_MATCH = internal constant %struct.MatchInfo zeroinitializer, align 8

define void @setup() {
  %pending = alloca %struct.MatchInfo, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %pending, ptr align 8 @NO_MATCH, i64 512, i1 false)
  %u = getelementptr inbounds %struct.MatchInfo, ptr %pending, i32 0, i32 2
  %v = load i64, ptr %u, align 8
  br label %done
done:
  ret void

  ; CHECK: movabsq $_NO_MATCH, {{.*}}
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
