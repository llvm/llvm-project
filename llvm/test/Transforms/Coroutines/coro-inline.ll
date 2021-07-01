; RUN: opt < %s -passes='always-inline,cgscc(coro-split)' -S | FileCheck %s
; RUN: opt < %s -sample-profile-file=%S/Inputs/sample.text.prof -pgo-kind=pgo-sample-use-pipeline -passes='sample-profile,cgscc(coro-split)' -disable-inlining=true -S | FileCheck %s

; Function Attrs: alwaysinline ssp uwtable
define void @ff() #0 {
entry:
  %id = call token @llvm.coro.id(i32 16, i8* null, i8* null, i8* null)
  %begin = call i8* @llvm.coro.begin(token %id, i8* null)
  ret void
}

; Function Attrs: alwaysinline ssp uwtable
define void @foo() #0 {
entry:
  %id1 = call token @llvm.coro.id(i32 16, i8* null, i8* null, i8* null)
  %begin = call i8* @llvm.coro.begin(token %id1, i8* null)
  call void @ff()
  ret void
}
; CHECK-LABEL: define void @foo()
; CHECK:         call void @ff()


declare token @llvm.coro.id(i32, i8* readnone, i8* nocapture readonly, i8*)
declare i8* @llvm.coro.begin(token, i8* writeonly)

attributes #0 = { alwaysinline ssp uwtable "coroutine.presplit"="1" "use-sample-profile" }

!llvm.dbg.cu = !{}
!llvm.module.flags = !{!1, !2, !3, !4}

!1 = !{i32 7, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"PIC Level", i32 2}
