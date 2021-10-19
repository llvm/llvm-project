; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -O1 < %s | FileCheck -check-prefixes=OPT,OPT-EXT %s
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes='default<O1>' < %s | FileCheck -check-prefixes=OPT,OPT-EXT %s
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -O1 --amdgpu-internalize-symbols < %s | FileCheck -check-prefixes=OPT,OPT-INT %s
; RUN: opt -S -mtriple=amdgcn-amd-amdhsa -passes='default<O1>' --amdgpu-internalize-symbols < %s | FileCheck -check-prefixes=OPT,OPT-INT %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefix=LLC %s

; OPT: declare void @foo4() local_unnamed_addr #0
; OPT: define internal fastcc void @0() unnamed_addr #1
; OPT-EXT: define void @foo3() local_unnamed_addr #1
; OPT-INT: define internal fastcc void @foo3.2() unnamed_addr #1
; OPT-EXT: define void @foo2() local_unnamed_addr #1
; OPT-INT: define internal fastcc void @foo2.3() unnamed_addr #1
; OPT-EXT: define void @foo1() local_unnamed_addr #1
; OPT-EXT:  tail call void @foo4()
; OPT-EXT:  tail call void @foo3()
; OPT-EXT:  tail call void @foo2()
; OPT-EXT:  tail call void @foo2()
; OPT-EXT:  tail call void @foo1()
; OPT-EXT:  tail call fastcc void @0()
; OPT-INT: define internal fastcc void @foo1.1() unnamed_addr #1
; OPT-INT:  tail call void @foo4()
; OPT-INT:  tail call fastcc void @foo3.2()
; OPT-INT:  tail call fastcc void @foo2.3()
; OPT-INT:  tail call fastcc void @foo2.3()
; OPT-INT:  tail call fastcc void @foo1.1()
; OPT-INT:  tail call fastcc void @0()
; OPT:      ret void
; OPT: define amdgpu_kernel void @kernel1() local_unnamed_addr #2
; OPT-EXT:  tail call fastcc void @foo1.1()
; OPT-INT:  tail call fastcc void @foo1()
; OPT:      ret void
; OPT: define amdgpu_kernel void @kernel2() local_unnamed_addr #3
; OPT-EXT:  tail call void @foo2()
; OPT-INT:  tail call fastcc void @foo2.3()
; OPT:      ret void
; OPT: define amdgpu_kernel void @kernel3() local_unnamed_addr #3
; OPT-EXT:  tail call void @foo1()
; OPT-INT:  tail call fastcc void @foo1.1()
; OPT:      ret void
; OPT-EXT: define internal fastcc void @foo1.1() unnamed_addr #4
; OPT-EXT:  tail call void @foo4()
; OPT-EXT:  tail call fastcc void @foo3.2()
; OPT-EXT:  tail call fastcc void @foo2.3()
; OPT-EXT:  tail call fastcc void @foo2.3()
; OPT-EXT:  tail call fastcc void @foo1.1()
; OPT-EXT:  tail call fastcc void @1()
; OPT-INT: define internal fastcc void @foo1() unnamed_addr #4
; OPT-INT:  tail call void @foo4()
; OPT-INT:  tail call fastcc void @foo3()
; OPT-INT:  tail call fastcc void @foo2()
; OPT-INT:  tail call fastcc void @foo2()
; OPT-INT:  tail call fastcc void @foo1()
; OPT-INT:  tail call fastcc void @1()
; OPT:      ret void
; OPT: define internal fastcc void @1() unnamed_addr #4
; OPT-EXT: define internal fastcc void @foo3.2() unnamed_addr #4
; OPT-INT: define internal fastcc void @foo3() unnamed_addr #4
; OPT-EXT: define internal fastcc void @foo2.3() unnamed_addr #4
; OPT-INT: define internal fastcc void @foo2() unnamed_addr #4
; OPT: attributes #0 = { {{.*}} "amdgpu-waves-per-eu"="1,1" "target-features"="+wavefrontsize64" }
; OPT: attributes #1 = { {{.*}} "target-features"="{{.*}},-wavefrontsize16,-wavefrontsize32,+wavefrontsize64{{.*}}" }
; OPT: attributes #2 = { {{.*}} "amdgpu-waves-per-eu"="2,4" "target-features"="+wavefrontsize32" }
; OPT: attributes #3 = { {{.*}} "target-features"="+wavefrontsize64" }
; OPT: attributes #4 = { {{.*}} "amdgpu-waves-per-eu"="2,4" "target-features"="{{.*}},-wavefrontsize16,+wavefrontsize32,-wavefrontsize64{{.*}}" }

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
; LLC: __unnamed_1@gotpcrel32@lo+4
; LLC: __unnamed_1@gotpcrel32@hi+12
; LLC: kernel1:
; LLC: foo1@gotpcrel32@lo+4
; LLC: foo1@gotpcrel32@hi+12
; LLC: kernel2:
; LLC: foo2@gotpcrel32@lo+4
; LLC: foo2@gotpcrel32@hi+12
; LLC: kernel3:
; LLC: foo1@gotpcrel32@lo+4
; LLC: foo1@gotpcrel32@hi+12

declare void @foo4() #1

define void @0() #1 {
entry:
  call void asm sideeffect "; sample asm", ""()
  ret void
}

define void @foo3() #4 {
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
  tail call void @0()
  ret void
}

define amdgpu_kernel void @kernel1() #0 {
entry:
  tail call void @foo1()
  ret void
}

define amdgpu_kernel void @kernel2() #2 {
entry:
  tail call void @foo2()
  ret void
}

define amdgpu_kernel void @kernel3() #3 {
entry:
  tail call void @foo1()
  ret void
}

attributes #0 = { nounwind "target-features"="+wavefrontsize32" "amdgpu-waves-per-eu"="2,4" }
attributes #1 = { noinline nounwind "target-features"="+wavefrontsize64" "amdgpu-waves-per-eu"="1,1" }
attributes #2 = { nounwind "target-features"="+wavefrontsize64" }
attributes #3 = { nounwind "target-features"="+wavefrontsize64" }
attributes #4 = { noinline nounwind "target-features"="+wavefrontsize64" "amdgpu-waves-per-eu"="2,4" }
