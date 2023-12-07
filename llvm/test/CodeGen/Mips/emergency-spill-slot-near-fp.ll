; RUN: llc -march=mipsel -O0 -relocation-model=pic < %s | FileCheck %s
; Check that register scavenging spill slot is close to $fp.
target triple="mipsel--"

; FIXME: After recent rework to FastISel, don't know how to trigger the
; emergency spill slot.  Filed PR48301.
; XFAIL: *
@var = external global i32
@ptrvar = external global ptr

; CHECK-LABEL: func:
define void @func() {
  %space = alloca i32, align 4
  %stackspace = alloca[16384 x i32], align 4

  ; ensure stackspace is not optimized out
  store volatile ptr %stackspace, ptr @ptrvar

  ; Load values to increase register pressure.
  %v0 = load volatile i32, ptr @var
  %v1 = load volatile i32, ptr @var
  %v2 = load volatile i32, ptr @var
  %v3 = load volatile i32, ptr @var
  %v4 = load volatile i32, ptr @var
  %v5 = load volatile i32, ptr @var
  %v6 = load volatile i32, ptr @var
  %v7 = load volatile i32, ptr @var
  %v8 = load volatile i32, ptr @var
  %v9 = load volatile i32, ptr @var
  %v10 = load volatile i32, ptr @var
  %v11 = load volatile i32, ptr @var
  %v12 = load volatile i32, ptr @var
  %v13 = load volatile i32, ptr @var
  %v14 = load volatile i32, ptr @var
  %v15 = load volatile i32, ptr @var
  %v16 = load volatile i32, ptr @var

  ; Computing a stack-relative values needs an additional register.
  ; We should get an emergency spill/reload for this.
  ; CHECK: sw ${{.*}}, 0($sp)
  ; CHECK: lw ${{.*}}, 0($sp)
  store volatile i32 %v0, ptr %space

  ; store values so they are used.
  store volatile i32 %v0, ptr @var
  store volatile i32 %v1, ptr @var
  store volatile i32 %v2, ptr @var
  store volatile i32 %v3, ptr @var
  store volatile i32 %v4, ptr @var
  store volatile i32 %v5, ptr @var
  store volatile i32 %v6, ptr @var
  store volatile i32 %v7, ptr @var
  store volatile i32 %v8, ptr @var
  store volatile i32 %v9, ptr @var
  store volatile i32 %v10, ptr @var
  store volatile i32 %v11, ptr @var
  store volatile i32 %v12, ptr @var
  store volatile i32 %v13, ptr @var
  store volatile i32 %v14, ptr @var
  store volatile i32 %v15, ptr @var
  store volatile i32 %v16, ptr @var

  ret void
}
