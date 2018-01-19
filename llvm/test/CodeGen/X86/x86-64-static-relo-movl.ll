; RUN: llc -mtriple=x86_64-pc-win32-macho -relocation-model=static -O0 < %s | FileCheck %s

; Ensure that we don't generate a movl and not a lea for a static relocation
; when compiling for 64 bit.

%struct.MatchInfo = type [64 x i64]

@NO_MATCH = internal constant %struct.MatchInfo zeroinitializer, align 8

define void @setup() {
  %pending = alloca %struct.MatchInfo, align 8
  %t = bitcast %struct.MatchInfo* %pending to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %t, i8* align 8 bitcast (%struct.MatchInfo* @NO_MATCH to i8*), i64 512, i1 false)
  %u = getelementptr inbounds %struct.MatchInfo, %struct.MatchInfo* %pending, i32 0, i32 2
  %v = load i64, i64* %u, align 8
  br label %done
done:
  ret void

  ; CHECK: movabsq $_NO_MATCH, {{.*}}
}

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i1)
