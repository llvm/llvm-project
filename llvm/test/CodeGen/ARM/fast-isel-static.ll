; RUN: llc < %s -mtriple=thumbv7-apple-darwin -O0 -relocation-model=static -arm-long-calls | FileCheck -check-prefix=LONG %s
; RUN: llc < %s -mtriple=thumbv7-apple-darwin -O0 -relocation-model=static | FileCheck -check-prefix=NORM %s

define void @myadd(float* %sum, float* %addend) nounwind {
entry:
  %sum.addr = alloca float*, align 4
  %addend.addr = alloca float*, align 4
  store float* %sum, float** %sum.addr, align 4
  store float* %addend, float** %addend.addr, align 4
  %tmp = load float** %sum.addr, align 4
  %tmp1 = load float* %tmp
  %tmp2 = load float** %addend.addr, align 4
  %tmp3 = load float* %tmp2
  %add = fadd float %tmp1, %tmp3
  %tmp4 = load float** %sum.addr, align 4
  store float %add, float* %tmp4
  ret void
}

define i32 @main(i32 %argc, i8** %argv) nounwind {
entry:
  %ztot = alloca float, align 4
  %z = alloca float, align 4
  store float 0.000000e+00, float* %ztot, align 4
  store float 1.000000e+00, float* %z, align 4
; CHECK-LONG: blx     r
; CHECK-NORM: bl      _myadd
  call void @myadd(float* %ztot, float* %z)
  ret i32 0
}
