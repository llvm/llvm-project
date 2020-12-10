; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -enable-misched=0 -post-RA-scheduler=0 -stress-regalloc=8 < %s | FileCheck %s -check-prefixes=GCN,MUBUF
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -enable-misched=0 -post-RA-scheduler=0 -stress-regalloc=8 -amdgpu-enable-flat-scratch < %s | FileCheck %s -check-prefixes=GCN,FLATSCR

; GCN-LABEL: spill_v2i32:
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:16 ; 4-byte Folded Spill
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:20 ; 4-byte Folded Spill
; FLATSCR:   scratch_store_dwordx2 off, v{{.*}} offset:16 ; 8-byte Folded Spill
; FLATSCR-NOT: scratch_store_dword
; GCN: ;;#ASMSTART
; GCN-NEXT: ;;#ASMEND
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:16 ; 4-byte Folded Reload
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:20 ; 4-byte Folded Reload
; FLATSCR:   scratch_load_dwordx2 v{{.*}} offset:16 ; 8-byte Folded Reload
; FLATSCR-NOT: scratch_load_dword

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

; GCN-LABEL: spill_v2f32:
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:16 ; 4-byte Folded Spill
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:20 ; 4-byte Folded Spill
; FLATSCR:   scratch_store_dwordx2 off, v{{.*}} offset:16 ; 8-byte Folded Spill
; FLATSCR-NOT: scratch_store_dword
; GCN: ;;#ASMSTART
; GCN-NEXT: ;;#ASMEND
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:16 ; 4-byte Folded Reload
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:20 ; 4-byte Folded Reload
; FLATSCR:   scratch_load_dwordx2 v{{.*}} offset:16 ; 8-byte Folded Reload
; FLATSCR-NOT: scratch_load_dword

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

; GCN-LABEL: spill_v3i32:
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:32 ; 4-byte Folded Spill
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:36 ; 4-byte Folded Spill
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:40 ; 4-byte Folded Spill
; FLATSCR:   scratch_store_dwordx3 off, v{{.*}} offset:32 ; 12-byte Folded Spill
; FLATSCR-NOT: scratch_store_dword
; GCN: ;;#ASMSTART
; GCN-NEXT: ;;#ASMEND
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:32 ; 4-byte Folded Reload
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:36 ; 4-byte Folded Reload
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:40 ; 4-byte Folded Reload
; FLATSCR:   scratch_load_dwordx3 v{{.*}} offset:32 ; 12-byte Folded Reload
; FLATSCR-NOT: scratch_load_dword

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

; GCN-LABEL: spill_v3f32:
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:32 ; 4-byte Folded Spill
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:36 ; 4-byte Folded Spill
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:40 ; 4-byte Folded Spill
; FLATSCR:   scratch_store_dwordx3 off, v{{.*}} offset:32 ; 12-byte Folded Spill
; FLATSCR-NOT: scratch_store_dword
; GCN: ;;#ASMSTART
; GCN-NEXT: ;;#ASMEND
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:32 ; 4-byte Folded Reload
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:36 ; 4-byte Folded Reload
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:40 ; 4-byte Folded Reload
; FLATSCR:   scratch_load_dwordx3 v{{.*}} offset:32 ; 12-byte Folded Reload
; FLATSCR-NOT: scratch_load_dword

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

; GCN-LABEL: spill_v4i32:
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:32 ; 4-byte Folded Spill
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:36 ; 4-byte Folded Spill
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:40 ; 4-byte Folded Spill
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:44 ; 4-byte Folded Spill
; FLATSCR:   scratch_store_dwordx4 off, v{{.*}} offset:32 ; 16-byte Folded Spill
; FLATSCR-NOT: scratch_store_dword
; GCN: ;;#ASMSTART
; GCN-NEXT: ;;#ASMEND
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:32 ; 4-byte Folded Reload
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:36 ; 4-byte Folded Reload
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:40 ; 4-byte Folded Reload
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:44 ; 4-byte Folded Reload
; FLATSCR:   scratch_load_dwordx4 v{{.*}} offset:32 ; 16-byte Folded Reload
; FLATSCR-NOT: scratch_load_dword

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

; GCN-LABEL: spill_v4f32:
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:32 ; 4-byte Folded Spill
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:36 ; 4-byte Folded Spill
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:40 ; 4-byte Folded Spill
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:44 ; 4-byte Folded Spill
; FLATSCR:   scratch_store_dwordx4 off, v{{.*}} offset:32 ; 16-byte Folded Spill
; FLATSCR-NOT: scratch_store_dword
; GCN: ;;#ASMSTART
; GCN-NEXT: ;;#ASMEND
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:32 ; 4-byte Folded Reload
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:36 ; 4-byte Folded Reload
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:40 ; 4-byte Folded Reload
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:44 ; 4-byte Folded Reload
; FLATSCR:   scratch_load_dwordx4 v{{.*}} offset:32 ; 16-byte Folded Reload
; FLATSCR-NOT: scratch_load_dword

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

; GCN-LABEL: spill_v5i32:
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:64 ; 4-byte Folded Spill
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:68 ; 4-byte Folded Spill
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:72 ; 4-byte Folded Spill
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:76 ; 4-byte Folded Spill
; FLATSCR-DAG: scratch_store_dwordx4 off, v{{.*}} offset:64 ; 16-byte Folded Spill
; FLATSCR-DAG: scratch_store_dword off, v{{.*}} offset:80 ; 4-byte Folded Spill
; FLATSCR-NOT: scratch_store_dword
; GCN: ;;#ASMSTART
; GCN-NEXT: ;;#ASMEND
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:64 ; 4-byte Folded Reload
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:68 ; 4-byte Folded Reload
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:72 ; 4-byte Folded Reload
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:76 ; 4-byte Folded Reload
; FLATSCR-DAG: scratch_load_dwordx4 v{{.*}} offset:64 ; 16-byte Folded Reload
; FLATSCR-DAG: scratch_load_dword v{{.*}} offset:80 ; 4-byte Folded Reload
; FLATSCR-NOT: scratch_load_dword
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

; GCN-LABEL: spill_v5f32:
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:64 ; 4-byte Folded Spill
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:68 ; 4-byte Folded Spill
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:72 ; 4-byte Folded Spill
; MUBUF-DAG: buffer_store_dword v{{.*}} offset:76 ; 4-byte Folded Spill
; FLATSCR-DAG: scratch_store_dwordx4 off, v{{.*}} offset:64 ; 16-byte Folded Spill
; FLATSCR-DAG: scratch_store_dword off, v{{.*}} offset:80 ; 4-byte Folded Spill
; FLATSCR-NOT: scratch_store_dword
; GCN: ;;#ASMSTART
; GCN-NEXT: ;;#ASMEND
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:64 ; 4-byte Folded Reload
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:68 ; 4-byte Folded Reload
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:72 ; 4-byte Folded Reload
; MUBUF-DAG: buffer_load_dword v{{.*}} offset:76 ; 4-byte Folded Reload
; FLATSCR-DAG: scratch_load_dwordx4 v{{.*}} offset:64 ; 16-byte Folded Reload
; FLATSCR-DAG: scratch_load_dword v{{.*}} offset:80 ; 4-byte Folded Reload
; FLATSCR-NOT: scratch_load_dword
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
