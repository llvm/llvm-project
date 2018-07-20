; RUN: llc -mtriple=x86_64-windows-gnu -exception-model=dwarf < %s | FileCheck %s

%struct.A = type { [4096 x i8] }

@a = common global i32 0, align 4
@b = common global i32 0, align 4

define void @fn1() nounwind uwtable {
entry:
  %ctx = alloca %struct.A, align 1
  %0 = load i32, i32* @a, align 4
  %tobool = icmp eq i32 %0, 0
  %div = sdiv i32 %0, 6
  %cond = select i1 %tobool, i32 %div, i32 %0
  store i32 %cond, i32* @b, align 4
  %1 = getelementptr inbounds %struct.A, %struct.A* %ctx, i64 0, i32 0, i64 0
  call void @llvm.lifetime.start.p0i8(i64 4096, i8* nonnull %1)
  %2 = ptrtoint %struct.A* %ctx to i64
  %3 = trunc i64 %2 to i32
  call void @fn2(i32 %3)
  call void @llvm.lifetime.end.p0i8(i64 4096, i8* nonnull %1)
  ret void
}

declare void @fn2(i32)
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture)
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture)

; CHECK-LABEL: fn1:
; CHECK: pushq %rax
; CHECK: movl $4128, %eax
; CHECK: callq ___chkstk_ms
; CHECK: subq %rax, %rsp
; CHECK: movq 4128(%rsp), %rax

; CHECK: addq $4136, %rsp
