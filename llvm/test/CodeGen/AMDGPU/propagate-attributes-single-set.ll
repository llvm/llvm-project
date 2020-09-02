; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -O1 < %s | FileCheck -check-prefix=OPT %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefix=LLC %s

; OPT: declare void @foo4() local_unnamed_addr #0
; OPT: define void @foo3() local_unnamed_addr #1
; OPT: define void @foo2() local_unnamed_addr #1
; OPT: define void @foo1() local_unnamed_addr #1
; OPT: define amdgpu_kernel void @kernel1() local_unnamed_addr #2
; OPT: define amdgpu_kernel void @kernel2() local_unnamed_addr #2
; OPT: attributes #0 = { {{.*}} "target-features"="+wavefrontsize64" }
; OPT: attributes #1 = { {{.*}} "target-features"="{{.*}},-wavefrontsize16,+wavefrontsize32,-wavefrontsize64
; OPT: attributes #2 = { {{.*}} "target-features"="+wavefrontsize32
; OPT: attributes #3 = { nounwind }

; LLC: foo3:
; LLC: sample asm
; LLC: foo2:
; LLC: sample asm
; LLC: foo1:
; LLC: foo4@gotpcrel32@lo+4
; LLC: foo4@gotpcrel32@hi+12
; LLC: foo3@gotpcrel32@lo+4
; LLC: foo3@gotpcrel32@hi+12
; LLC: foo2@gotpcrel32@lo+4
; LLC: foo2@gotpcrel32@hi+12
; LLC: foo1@gotpcrel32@lo+4
; LLC: foo1@gotpcrel32@hi+12
; LLC: kernel1:
; LLC: foo1@gotpcrel32@lo+4
; LLC: foo1@gotpcrel32@hi+12
; LLC: kernel2:
; LLC: foo2@gotpcrel32@lo+4
; LLC: foo2@gotpcrel32@hi+12

declare void @foo4() #1

define void @foo3() #1 {
entry:
  call void asm sideeffect "; sample asm", ""()
  ret void
}

define void @foo2() #1 {
entry:
  call void asm sideeffect "; sample asm", ""()
  ret void
}

define void @foo1() #1 {
entry:
  tail call void @foo4()
  tail call void @foo3()
  tail call void @foo2()
  tail call void @foo2()
  tail call void @foo1()
  ret void
}

define amdgpu_kernel void @kernel1() #0 {
entry:
  tail call void @foo1()
  ret void
}

define amdgpu_kernel void @kernel2() #0 {
entry:
  tail call void @foo2()
  ret void
}

attributes #0 = { nounwind "target-features"="+wavefrontsize32" }
attributes #1 = { noinline nounwind "target-features"="+wavefrontsize64" }
