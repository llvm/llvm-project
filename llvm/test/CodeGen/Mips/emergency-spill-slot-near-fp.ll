; RUN: llc -march=mipsel -O0 -relocation-model=pic < %s | FileCheck %s
; Check that register scavenging spill slot is close to $fp.
target triple="mipsel--"

@var = external global i32
@ptrvar = external global i8*

; CHECK-LABEL: func:
define void @func() {
  %space = alloca i32, align 4
  %stackspace = alloca[16384 x i32], align 4

  ; ensure stackspace is not optimized out
  %stackspace_casted = bitcast [16384 x i32]* %stackspace to i8*
  store volatile i8* %stackspace_casted, i8** @ptrvar

  ; Load values to increase register pressure.
  %v0 = load volatile i32, i32* @var
  %v1 = load volatile i32, i32* @var
  %v2 = load volatile i32, i32* @var
  %v3 = load volatile i32, i32* @var
  %v4 = load volatile i32, i32* @var
  %v5 = load volatile i32, i32* @var
  %v6 = load volatile i32, i32* @var
  %v7 = load volatile i32, i32* @var
  %v8 = load volatile i32, i32* @var
  %v9 = load volatile i32, i32* @var
  %v10 = load volatile i32, i32* @var
  %v11 = load volatile i32, i32* @var
  %v12 = load volatile i32, i32* @var
  %v13 = load volatile i32, i32* @var
  %v14 = load volatile i32, i32* @var
  %v15 = load volatile i32, i32* @var
  %v16 = load volatile i32, i32* @var

  ; Computing a stack-relative values needs an additional register.
  ; We should get an emergency spill/reload for this.
  ; CHECK: sw ${{.*}}, 0($sp)
  ; CHECK: lw ${{.*}}, 0($sp)
  store volatile i32 %v0, i32* %space

  ; store values so they are used.
  store volatile i32 %v0, i32* @var
  store volatile i32 %v1, i32* @var
  store volatile i32 %v2, i32* @var
  store volatile i32 %v3, i32* @var
  store volatile i32 %v4, i32* @var
  store volatile i32 %v5, i32* @var
  store volatile i32 %v6, i32* @var
  store volatile i32 %v7, i32* @var
  store volatile i32 %v8, i32* @var
  store volatile i32 %v9, i32* @var
  store volatile i32 %v10, i32* @var
  store volatile i32 %v11, i32* @var
  store volatile i32 %v12, i32* @var
  store volatile i32 %v13, i32* @var
  store volatile i32 %v14, i32* @var
  store volatile i32 %v15, i32* @var
  store volatile i32 %v16, i32* @var

  ret void
}
