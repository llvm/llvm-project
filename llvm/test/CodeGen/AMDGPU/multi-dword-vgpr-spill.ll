; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -enable-misched=0 -post-RA-scheduler=0 -stress-regalloc=8 < %s | FileCheck %s

; CHECK-LABEL: spill_v2i32:
; CHECK-DAG: buffer_store_dword v{{.*}} offset:24 ; 4-byte Folded Spill
; CHECK-DAG: buffer_store_dword v{{.*}} offset:28 ; 4-byte Folded Spill
; CHECK: ;;#ASMSTART
; CHECK-NEXT: ;;#ASMEND
; CHECK-DAG: buffer_load_dword v{{.*}} offset:24 ; 4-byte Folded Reload
; CHECK-DAG: buffer_load_dword v{{.*}} offset:28 ; 4-byte Folded Reload

define void @spill_v2i32() {
entry:
  %alloca = alloca <2 x i32>, i32 2, align 4, addrspace(5)

  %aptr = getelementptr <2 x i32>, <2 x i32> addrspace(5)* %alloca, i32 1
  %a = load volatile <2 x i32>, <2 x i32> addrspace(5)* %aptr

  ; Force %a to spill.
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}" ()

  %outptr = getelementptr <2 x i32>, <2 x i32> addrspace(5)* %alloca, i32 1
  store volatile <2 x i32> %a, <2 x i32> addrspace(5)* %outptr

  ret void
}

; CHECK-LABEL: spill_v2f32:
; CHECK-DAG: buffer_store_dword v{{.*}} offset:24 ; 4-byte Folded Spill
; CHECK-DAG: buffer_store_dword v{{.*}} offset:28 ; 4-byte Folded Spill
; CHECK: ;;#ASMSTART
; CHECK-NEXT: ;;#ASMEND
; CHECK-DAG: buffer_load_dword v{{.*}} offset:24 ; 4-byte Folded Reload
; CHECK-DAG: buffer_load_dword v{{.*}} offset:28 ; 4-byte Folded Reload

define void @spill_v2f32() {
entry:
  %alloca = alloca <2 x i32>, i32 2, align 4, addrspace(5)

  %aptr = getelementptr <2 x i32>, <2 x i32> addrspace(5)* %alloca, i32 1
  %a = load volatile <2 x i32>, <2 x i32> addrspace(5)* %aptr

  ; Force %a to spill.
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}" ()

  %outptr = getelementptr <2 x i32>, <2 x i32> addrspace(5)* %alloca, i32 1
  store volatile <2 x i32> %a, <2 x i32> addrspace(5)* %outptr

  ret void
}

; CHECK-LABEL: spill_v3i32:
; CHECK-DAG: buffer_store_dword v{{.*}} offset:48 ; 4-byte Folded Spill
; CHECK-DAG: buffer_store_dword v{{.*}} offset:52 ; 4-byte Folded Spill
; CHECK-DAG: buffer_store_dword v{{.*}} offset:56 ; 4-byte Folded Spill
; CHECK: ;;#ASMSTART
; CHECK-NEXT: ;;#ASMEND
; CHECK-DAG: buffer_load_dword v{{.*}} offset:48 ; 4-byte Folded Reload
; CHECK-DAG: buffer_load_dword v{{.*}} offset:52 ; 4-byte Folded Reload
; CHECK-DAG: buffer_load_dword v{{.*}} offset:56 ; 4-byte Folded Reload

define void @spill_v3i32() {
entry:
  %alloca = alloca <3 x i32>, i32 2, align 4, addrspace(5)

  %aptr = getelementptr <3 x i32>, <3 x i32> addrspace(5)* %alloca, i32 1
  %a = load volatile <3 x i32>, <3 x i32> addrspace(5)* %aptr

  ; Force %a to spill.
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}" ()

  %outptr = getelementptr <3 x i32>, <3 x i32> addrspace(5)* %alloca, i32 1
  store volatile <3 x i32> %a, <3 x i32> addrspace(5)* %outptr

  ret void
}

; CHECK-LABEL: spill_v3f32:
; CHECK-DAG: buffer_store_dword v{{.*}} offset:48 ; 4-byte Folded Spill
; CHECK-DAG: buffer_store_dword v{{.*}} offset:52 ; 4-byte Folded Spill
; CHECK-DAG: buffer_store_dword v{{.*}} offset:56 ; 4-byte Folded Spill
; CHECK: ;;#ASMSTART
; CHECK-NEXT: ;;#ASMEND
; CHECK-DAG: buffer_load_dword v{{.*}} offset:48 ; 4-byte Folded Reload
; CHECK-DAG: buffer_load_dword v{{.*}} offset:52 ; 4-byte Folded Reload
; CHECK-DAG: buffer_load_dword v{{.*}} offset:56 ; 4-byte Folded Reload

define void @spill_v3f32() {
entry:
  %alloca = alloca <3 x i32>, i32 2, align 4, addrspace(5)

  %aptr = getelementptr <3 x i32>, <3 x i32> addrspace(5)* %alloca, i32 1
  %a = load volatile <3 x i32>, <3 x i32> addrspace(5)* %aptr

  ; Force %a to spill.
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}" ()

  %outptr = getelementptr <3 x i32>, <3 x i32> addrspace(5)* %alloca, i32 1
  store volatile <3 x i32> %a, <3 x i32> addrspace(5)* %outptr

  ret void
}

; CHECK-LABEL: spill_v4i32:
; CHECK-DAG: buffer_store_dword v{{.*}} offset:48 ; 4-byte Folded Spill
; CHECK-DAG: buffer_store_dword v{{.*}} offset:52 ; 4-byte Folded Spill
; CHECK-DAG: buffer_store_dword v{{.*}} offset:56 ; 4-byte Folded Spill
; CHECK-DAG: buffer_store_dword v{{.*}} offset:60 ; 4-byte Folded Spill
; CHECK: ;;#ASMSTART
; CHECK-NEXT: ;;#ASMEND
; CHECK-DAG: buffer_load_dword v{{.*}} offset:48 ; 4-byte Folded Reload
; CHECK-DAG: buffer_load_dword v{{.*}} offset:52 ; 4-byte Folded Reload
; CHECK-DAG: buffer_load_dword v{{.*}} offset:56 ; 4-byte Folded Reload
; CHECK-DAG: buffer_load_dword v{{.*}} offset:60 ; 4-byte Folded Reload

define void @spill_v4i32() {
entry:
  %alloca = alloca <4 x i32>, i32 2, align 4, addrspace(5)

  %aptr = getelementptr <4 x i32>, <4 x i32> addrspace(5)* %alloca, i32 1
  %a = load volatile <4 x i32>, <4 x i32> addrspace(5)* %aptr

  ; Force %a to spill.
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}" ()

  %outptr = getelementptr <4 x i32>, <4 x i32> addrspace(5)* %alloca, i32 1
  store volatile <4 x i32> %a, <4 x i32> addrspace(5)* %outptr

  ret void
}

; CHECK-LABEL: spill_v4f32:
; CHECK-DAG: buffer_store_dword v{{.*}} offset:48 ; 4-byte Folded Spill
; CHECK-DAG: buffer_store_dword v{{.*}} offset:52 ; 4-byte Folded Spill
; CHECK-DAG: buffer_store_dword v{{.*}} offset:56 ; 4-byte Folded Spill
; CHECK-DAG: buffer_store_dword v{{.*}} offset:60 ; 4-byte Folded Spill
; CHECK: ;;#ASMSTART
; CHECK-NEXT: ;;#ASMEND
; CHECK-DAG: buffer_load_dword v{{.*}} offset:48 ; 4-byte Folded Reload
; CHECK-DAG: buffer_load_dword v{{.*}} offset:52 ; 4-byte Folded Reload
; CHECK-DAG: buffer_load_dword v{{.*}} offset:56 ; 4-byte Folded Reload
; CHECK-DAG: buffer_load_dword v{{.*}} offset:60 ; 4-byte Folded Reload

define void @spill_v4f32() {
entry:
  %alloca = alloca <4 x i32>, i32 2, align 4, addrspace(5)

  %aptr = getelementptr <4 x i32>, <4 x i32> addrspace(5)* %alloca, i32 1
  %a = load volatile <4 x i32>, <4 x i32> addrspace(5)* %aptr

  ; Force %a to spill.
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}" ()

  %outptr = getelementptr <4 x i32>, <4 x i32> addrspace(5)* %alloca, i32 1
  store volatile <4 x i32> %a, <4 x i32> addrspace(5)* %outptr

  ret void
}

; CHECK-LABEL: spill_v5i32:
; CHECK-DAG: buffer_store_dword v{{.*}} offset:96 ; 4-byte Folded Spill
; CHECK-DAG: buffer_store_dword v{{.*}} offset:100 ; 4-byte Folded Spill
; CHECK-DAG: buffer_store_dword v{{.*}} offset:104 ; 4-byte Folded Spill
; CHECK-DAG: buffer_store_dword v{{.*}} offset:108 ; 4-byte Folded Spill
; CHECK: ;;#ASMSTART
; CHECK-NEXT: ;;#ASMEND
; CHECK-DAG: buffer_load_dword v{{.*}} offset:96 ; 4-byte Folded Reload
; CHECK-DAG: buffer_load_dword v{{.*}} offset:100 ; 4-byte Folded Reload
; CHECK-DAG: buffer_load_dword v{{.*}} offset:104 ; 4-byte Folded Reload
; CHECK-DAG: buffer_load_dword v{{.*}} offset:108 ; 4-byte Folded Reload

define void @spill_v5i32() {
entry:
  %alloca = alloca <5 x i32>, i32 2, align 4, addrspace(5)

  %aptr = getelementptr <5 x i32>, <5 x i32> addrspace(5)* %alloca, i32 1
  %a = load volatile <5 x i32>, <5 x i32> addrspace(5)* %aptr

  ; Force %a to spill.
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}" ()

  %outptr = getelementptr <5 x i32>, <5 x i32> addrspace(5)* %alloca, i32 1
  store volatile <5 x i32> %a, <5 x i32> addrspace(5)* %outptr

  ret void
}

; CHECK-LABEL: spill_v5f32:
; CHECK-DAG: buffer_store_dword v{{.*}} offset:96 ; 4-byte Folded Spill
; CHECK-DAG: buffer_store_dword v{{.*}} offset:100 ; 4-byte Folded Spill
; CHECK-DAG: buffer_store_dword v{{.*}} offset:104 ; 4-byte Folded Spill
; CHECK-DAG: buffer_store_dword v{{.*}} offset:108 ; 4-byte Folded Spill
; CHECK: ;;#ASMSTART
; CHECK-NEXT: ;;#ASMEND
; CHECK-DAG: buffer_load_dword v{{.*}} offset:96 ; 4-byte Folded Reload
; CHECK-DAG: buffer_load_dword v{{.*}} offset:100 ; 4-byte Folded Reload
; CHECK-DAG: buffer_load_dword v{{.*}} offset:104 ; 4-byte Folded Reload
; CHECK-DAG: buffer_load_dword v{{.*}} offset:108 ; 4-byte Folded Reload

define void @spill_v5f32() {
entry:
  %alloca = alloca <5 x i32>, i32 2, align 4, addrspace(5)

  %aptr = getelementptr <5 x i32>, <5 x i32> addrspace(5)* %alloca, i32 1
  %a = load volatile <5 x i32>, <5 x i32> addrspace(5)* %aptr

  ; Force %a to spill.
  call void asm sideeffect "", "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7}" ()

  %outptr = getelementptr <5 x i32>, <5 x i32> addrspace(5)* %alloca, i32 1
  store volatile <5 x i32> %a, <5 x i32> addrspace(5)* %outptr

  ret void
}



