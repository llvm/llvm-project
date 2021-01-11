; RUN: llc -O0 -fast-isel -fast-isel-abort=1 -verify-machineinstrs -mtriple=arm64-apple-darwin < %s | FileCheck %s

define void @t0(i32 %a) nounwind {
entry:
; CHECK-LABEL: t0:
; CHECK: str {{w[0-9]+}}, [sp, #12]
; CHECK-NEXT: ldr [[REGISTER:w[0-9]+]], [sp, #12]
; CHECK-NEXT: str [[REGISTER]], [sp, #12]
; CHECK: ret
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr
  %tmp = load i32, i32* %a.addr
  store i32 %tmp, i32* %a.addr
  ret void
}

define void @t1(i64 %a) nounwind {
; CHECK-LABEL: t1:
; CHECK: str {{x[0-9]+}}, [sp, #8]
; CHECK-NEXT: ldr [[REGISTER:x[0-9]+]], [sp, #8]
; CHECK-NEXT: str [[REGISTER]], [sp, #8]
; CHECK: ret
  %a.addr = alloca i64, align 4
  store i64 %a, i64* %a.addr
  %tmp = load i64, i64* %a.addr
  store i64 %tmp, i64* %a.addr
  ret void
}

define zeroext i1 @i1(i1 %a) nounwind {
entry:
; CHECK-LABEL: i1:
; CHECK: and [[REG:w[0-9]+]], w0, #0x1
; CHECK: strb [[REG]], [sp, #15]
; CHECK: ldrb [[REG1:w[0-9]+]], [sp, #15]
; CHECK: and [[REG2:w[0-9]+]], [[REG1]], #0x1
; CHECK: and w0, [[REG2]], #0x1
; CHECK: add sp, sp, #16
; CHECK: ret
  %a.addr = alloca i1, align 1
  store i1 %a, i1* %a.addr, align 1
  %0 = load i1, i1* %a.addr, align 1
  ret i1 %0
}

define i32 @t2(i32 *%ptr) nounwind {
entry:
; CHECK-LABEL: t2:
; CHECK: ldur w0, [x0, #-4]
; CHECK: ret
  %0 = getelementptr i32, i32 *%ptr, i32 -1
  %1 = load i32, i32* %0, align 4
  ret i32 %1
}

define i32 @t3(i32 *%ptr) nounwind {
entry:
; CHECK-LABEL: t3:
; CHECK: ldur w0, [x0, #-256]
; CHECK: ret
  %0 = getelementptr i32, i32 *%ptr, i32 -64
  %1 = load i32, i32* %0, align 4
  ret i32 %1
}

define void @t4(i32 *%ptr) nounwind {
entry:
; CHECK-LABEL: t4:
; CHECK: stur wzr, [x0, #-4]
; CHECK: ret
  %0 = getelementptr i32, i32 *%ptr, i32 -1
  store i32 0, i32* %0, align 4
  ret void
}

define void @t5(i32 *%ptr) nounwind {
entry:
; CHECK-LABEL: t5:
; CHECK: stur wzr, [x0, #-256]
; CHECK: ret
  %0 = getelementptr i32, i32 *%ptr, i32 -64
  store i32 0, i32* %0, align 4
  ret void
}

define void @t6() nounwind {
; CHECK-LABEL: t6:
; CHECK: brk #0x1
  tail call void @llvm.trap()
  ret void
}

declare void @llvm.trap() nounwind

define void @ands(i32* %addr) {
; FIXME: 'select i1 undef' makes this unreliable (ub?).
; COM: CHECK-LABEL: ands:
; COM: CHECK: tst [[COND:w[0-9]+]], #0x1
; COM: CHECK-NEXT: mov w{{[0-9]+}}, #2
; COM: CHECK-NEXT: mov w{{[0-9]+}}, #1
; COM: CHECK-NEXT: csel [[COND]],
entry:
  %cond91 = select i1 undef, i32 1, i32 2
  store i32 %cond91, i32* %addr, align 4
  ret void
}

define i64 @mul_umul(i64 %arg) {
; CHECK-LABEL: mul_umul:
; CHECK: mul x{{[0-9]+}}, [[ARG1:x[0-9]+]], [[ARG2:x[0-9]+]]
; CHECK-NEXT: umulh x{{[0-9]+}}, [[ARG1]], [[ARG2]]
entry:
  %sub.ptr.div = sdiv exact i64 %arg, 8
  %tmp = call { i64, i1 } @llvm.umul.with.overflow.i64(i64 %sub.ptr.div, i64 8)
  %tmp1 = extractvalue { i64, i1 } %tmp, 0
  ret i64 %tmp1
}

declare { i64, i1 } @llvm.umul.with.overflow.i64(i64, i64)

define void @logicalReg() {
; Make sure we generate a logical reg = reg, reg instruction without any
; machine verifier errors.
; CHECK-LABEL: logicalReg:
; CHECK: orr w{{[0-9]+}}, w{{[0-9]+}}, w{{[0-9]+}}
; CHECK: ret
entry:
  br i1 undef, label %cond.end, label %cond.false

cond.false:
  %cond = select i1 undef, i1 true, i1 false
  br label %cond.end

cond.end:
  %cond13 = phi i1 [ %cond, %cond.false ], [ true, %entry ]
  ret void
}

