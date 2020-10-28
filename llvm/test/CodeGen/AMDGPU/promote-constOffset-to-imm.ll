; RUN: llc -mtriple=amdgcn -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX8 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX9 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10 %s

declare i64 @_Z13get_global_idj(i32)

define amdgpu_kernel void @clmem_read_simplified(i8 addrspace(1)*  %buffer) {
; GCN-LABEL: clmem_read_simplified:
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
;
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:2048
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:2048
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:2048
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:-4096
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:2048
;
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:-2048
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:-2048
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:-2048
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}

entry:
  %call = tail call i64 @_Z13get_global_idj(i32 0)
  %conv = and i64 %call, 255
  %a0 = shl i64 %call, 7
  %idx.ext11 = and i64 %a0, 4294934528
  %add.ptr12 = getelementptr inbounds i8, i8 addrspace(1)* %buffer, i64 %idx.ext11
  %saddr = bitcast i8 addrspace(1)* %add.ptr12 to i64 addrspace(1)*

  %addr1 = getelementptr inbounds i64, i64 addrspace(1)* %saddr, i64 %conv
  %load1 = load i64, i64 addrspace(1)* %addr1, align 8
  %addr2 = getelementptr inbounds i64, i64 addrspace(1)* %addr1, i64 256
  %load2 = load i64, i64 addrspace(1)* %addr2, align 8
  %add.1 = add i64 %load2, %load1

  %add.ptr8.2 = getelementptr inbounds i64, i64 addrspace(1)* %addr1, i64 512
  %load3 = load i64, i64 addrspace(1)* %add.ptr8.2, align 8
  %add.2 = add i64 %load3, %add.1
  %add.ptr8.3 = getelementptr inbounds i64, i64 addrspace(1)* %addr1, i64 768
  %load4 = load i64, i64 addrspace(1)* %add.ptr8.3, align 8
  %add.3 = add i64 %load4, %add.2

  %add.ptr8.4 = getelementptr inbounds i64, i64 addrspace(1)* %addr1, i64 1024
  %load5 = load i64, i64 addrspace(1)* %add.ptr8.4, align 8
  %add.4 = add i64 %load5, %add.3
  %add.ptr8.5 = getelementptr inbounds i64, i64 addrspace(1)* %addr1, i64 1280
  %load6 = load i64, i64 addrspace(1)* %add.ptr8.5, align 8
  %add.5 = add i64 %load6, %add.4

  %add.ptr8.6 = getelementptr inbounds i64, i64 addrspace(1)* %addr1, i64 1536
  %load7 = load i64, i64 addrspace(1)* %add.ptr8.6, align 8
  %add.6 = add i64 %load7, %add.5
  %add.ptr8.7 = getelementptr inbounds i64, i64 addrspace(1)* %addr1, i64 1792
  %load8 = load i64, i64 addrspace(1)* %add.ptr8.7, align 8
  %add.7 = add i64 %load8, %add.6

  store i64 %add.7, i64 addrspace(1)* %saddr, align 8
  ret void
}

define hidden amdgpu_kernel void @clmem_read(i8 addrspace(1)*  %buffer) {
; GCN-LABEL: clmem_read:
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
;
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:-4096
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:-2048
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:-2048
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:-2048
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:-2048
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:-4096
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:-2048
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
;
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:-2048
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:-2048
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:-2048
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:-2048
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:-2048
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
entry:
  %call = tail call i64 @_Z13get_global_idj(i32 0)
  %conv = and i64 %call, 255
  %a0 = shl i64 %call, 17
  %idx.ext11 = and i64 %a0, 4261412864
  %add.ptr12 = getelementptr inbounds i8, i8 addrspace(1)* %buffer, i64 %idx.ext11
  %a1 = bitcast i8 addrspace(1)* %add.ptr12 to i64 addrspace(1)*
  %add.ptr6 = getelementptr inbounds i64, i64 addrspace(1)* %a1, i64 %conv
  br label %for.cond.preheader

while.cond.loopexit:                              ; preds = %for.body
  %dec = add nsw i32 %dec31, -1
  %tobool = icmp eq i32 %dec31, 0
  br i1 %tobool, label %while.end, label %for.cond.preheader

for.cond.preheader:                               ; preds = %entry, %while.cond.loopexit
  %dec31 = phi i32 [ 127, %entry ], [ %dec, %while.cond.loopexit ]
  %sum.030 = phi i64 [ 0, %entry ], [ %add.10, %while.cond.loopexit ]
  br label %for.body

for.body:                                         ; preds = %for.body, %for.cond.preheader
  %block.029 = phi i32 [ 0, %for.cond.preheader ], [ %add9.31, %for.body ]
  %sum.128 = phi i64 [ %sum.030, %for.cond.preheader ], [ %add.10, %for.body ]
  %conv3 = zext i32 %block.029 to i64
  %add.ptr8 = getelementptr inbounds i64, i64 addrspace(1)* %add.ptr6, i64 %conv3
  %load1 = load i64, i64 addrspace(1)* %add.ptr8, align 8
  %add = add i64 %load1, %sum.128

  %add9 = or i32 %block.029, 256
  %conv3.1 = zext i32 %add9 to i64
  %add.ptr8.1 = getelementptr inbounds i64, i64 addrspace(1)* %add.ptr6, i64 %conv3.1
  %load2 = load i64, i64 addrspace(1)* %add.ptr8.1, align 8
  %add.1 = add i64 %load2, %add

  %add9.1 = or i32 %block.029, 512
  %conv3.2 = zext i32 %add9.1 to i64
  %add.ptr8.2 = getelementptr inbounds i64, i64 addrspace(1)* %add.ptr6, i64 %conv3.2
  %l3 = load i64, i64 addrspace(1)* %add.ptr8.2, align 8
  %add.2 = add i64 %l3, %add.1

  %add9.2 = or i32 %block.029, 768
  %conv3.3 = zext i32 %add9.2 to i64
  %add.ptr8.3 = getelementptr inbounds i64, i64 addrspace(1)* %add.ptr6, i64 %conv3.3
  %l4 = load i64, i64 addrspace(1)* %add.ptr8.3, align 8
  %add.3 = add i64 %l4, %add.2

  %add9.3 = or i32 %block.029, 1024
  %conv3.4 = zext i32 %add9.3 to i64
  %add.ptr8.4 = getelementptr inbounds i64, i64 addrspace(1)* %add.ptr6, i64 %conv3.4
  %l5 = load i64, i64 addrspace(1)* %add.ptr8.4, align 8
  %add.4 = add i64 %l5, %add.3

  %add9.4 = or i32 %block.029, 1280
  %conv3.5 = zext i32 %add9.4 to i64
  %add.ptr8.5 = getelementptr inbounds i64, i64 addrspace(1)* %add.ptr6, i64 %conv3.5
  %l6 = load i64, i64 addrspace(1)* %add.ptr8.5, align 8
  %add.5 = add i64 %l6, %add.4

  %add9.5 = or i32 %block.029, 1536
  %conv3.6 = zext i32 %add9.5 to i64
  %add.ptr8.6 = getelementptr inbounds i64, i64 addrspace(1)* %add.ptr6, i64 %conv3.6
  %load7 = load i64, i64 addrspace(1)* %add.ptr8.6, align 8
  %add.6 = add i64 %load7, %add.5

  %add9.6 = or i32 %block.029, 1792
  %conv3.7 = zext i32 %add9.6 to i64
  %add.ptr8.7 = getelementptr inbounds i64, i64 addrspace(1)* %add.ptr6, i64 %conv3.7
  %load8 = load i64, i64 addrspace(1)* %add.ptr8.7, align 8
  %add.7 = add i64 %load8, %add.6

  %add9.7 = or i32 %block.029, 2048
  %conv3.8 = zext i32 %add9.7 to i64
  %add.ptr8.8 = getelementptr inbounds i64, i64 addrspace(1)* %add.ptr6, i64 %conv3.8
  %load9 = load i64, i64 addrspace(1)* %add.ptr8.8, align 8
  %add.8 = add i64 %load9, %add.7

  %add9.8 = or i32 %block.029, 2304
  %conv3.9 = zext i32 %add9.8 to i64
  %add.ptr8.9 = getelementptr inbounds i64, i64 addrspace(1)* %add.ptr6, i64 %conv3.9
  %load10 = load i64, i64 addrspace(1)* %add.ptr8.9, align 8
  %add.9 = add i64 %load10, %add.8

  %add9.9 = or i32 %block.029, 2560
  %conv3.10 = zext i32 %add9.9 to i64
  %add.ptr8.10 = getelementptr inbounds i64, i64 addrspace(1)* %add.ptr6, i64 %conv3.10
  %load11 = load i64, i64 addrspace(1)* %add.ptr8.10, align 8
  %add.10 = add i64 %load11, %add.9

  %add9.31 = add nuw nsw i32 %block.029, 8192
  %cmp.31 = icmp ult i32 %add9.31, 4194304
  br i1 %cmp.31, label %for.body, label %while.cond.loopexit

while.end:                                        ; preds = %while.cond.loopexit
  store i64 %add.10, i64 addrspace(1)* %a1, align 8
  ret void
}

; using 32bit address.
define amdgpu_kernel void @Address32(i8 addrspace(1)* %buffer) {
; GCN-LABEL: Address32:
; GFX8:    flat_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
;
; GFX9:    global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX9:    global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off offset:1024
; GFX9:    global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off offset:2048
; GFX9:    global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off offset:3072
; GFX9:    global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX9:    global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off offset:1024
; GFX9:    global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off offset:2048
; GFX9:    global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off offset:3072
; GFX9:    global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX9:    global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off offset:1024
;
; GFX10:   global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off offset:1024
; GFX10:   global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off offset:1024
; GFX10:   global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off offset:-2048
; GFX10:   global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off offset:1024
; GFX10:   global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off offset:1024
; GFX10:   global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off offset:-2048
; GFX10:   global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off offset:1024
entry:
   %call = tail call i64 @_Z13get_global_idj(i32 0)
   %conv = and i64 %call, 255
   %id = shl i64 %call, 7
   %idx.ext11 = and i64 %id, 4294934528
   %add.ptr12 = getelementptr inbounds i8, i8 addrspace(1)* %buffer, i64 %idx.ext11
   %addr = bitcast i8 addrspace(1)* %add.ptr12 to i32 addrspace(1)*

   %add.ptr6 = getelementptr inbounds i32, i32 addrspace(1)* %addr, i64 %conv
   %load1 = load i32, i32 addrspace(1)* %add.ptr6, align 4

   %add.ptr8.1 = getelementptr inbounds i32, i32 addrspace(1)* %add.ptr6, i64 256
   %load2 = load i32, i32 addrspace(1)* %add.ptr8.1, align 4
   %add.1 = add i32 %load2, %load1

   %add.ptr8.2 = getelementptr inbounds i32, i32 addrspace(1)* %add.ptr6, i64 512
   %load3 = load i32, i32 addrspace(1)* %add.ptr8.2, align 4
   %add.2 = add i32 %load3, %add.1

   %add.ptr8.3 = getelementptr inbounds i32, i32 addrspace(1)* %add.ptr6, i64 768
   %load4 = load i32, i32 addrspace(1)* %add.ptr8.3, align 4
   %add.3 = add i32 %load4, %add.2

   %add.ptr8.4 = getelementptr inbounds i32, i32 addrspace(1)* %add.ptr6, i64 1024
   %load5 = load i32, i32 addrspace(1)* %add.ptr8.4, align 4
   %add.4 = add i32 %load5, %add.3

   %add.ptr8.5 = getelementptr inbounds i32, i32 addrspace(1)* %add.ptr6, i64 1280
   %load6 = load i32, i32 addrspace(1)* %add.ptr8.5, align 4
   %add.5 = add i32 %load6, %add.4

   %add.ptr8.6 = getelementptr inbounds i32, i32 addrspace(1)* %add.ptr6, i64 1536
   %load7 = load i32, i32 addrspace(1)* %add.ptr8.6, align 4
   %add.6 = add i32 %load7, %add.5

   %add.ptr8.7 = getelementptr inbounds i32, i32 addrspace(1)* %add.ptr6, i64 1792
   %load8 = load i32, i32 addrspace(1)* %add.ptr8.7, align 4
   %add.7 = add i32 %load8, %add.6

   %add.ptr8.8 = getelementptr inbounds i32, i32 addrspace(1)* %add.ptr6, i64 2048
   %load9 = load i32, i32 addrspace(1)* %add.ptr8.8, align 4
   %add.8 = add i32 %load9, %add.7

   %add.ptr8.9 = getelementptr inbounds i32, i32 addrspace(1)* %add.ptr6, i64 2304
   %load10 = load i32, i32 addrspace(1)* %add.ptr8.9, align 4
   %add.9 = add i32 %load10, %add.8

   store i32 %add.9, i32 addrspace(1)* %addr, align 4
   ret void
}

define amdgpu_kernel void @Offset64(i8 addrspace(1)*  %buffer) {
; GCN-LABEL: Offset64:
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
;
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:-4096
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:2048
;
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:-2048
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
entry:
  %call = tail call i64 @_Z13get_global_idj(i32 0)
  %conv = and i64 %call, 255
  %a0 = shl i64 %call, 7
  %idx.ext11 = and i64 %a0, 4294934528
  %add.ptr12 = getelementptr inbounds i8, i8 addrspace(1)* %buffer, i64 %idx.ext11
  %saddr = bitcast i8 addrspace(1)* %add.ptr12 to i64 addrspace(1)*

  %addr1 = getelementptr inbounds i64, i64 addrspace(1)* %saddr, i64 %conv
  %load1 = load i64, i64 addrspace(1)* %addr1, align 8

  %addr2 = getelementptr inbounds i64, i64 addrspace(1)* %addr1, i64 536870400
  %load2 = load i64, i64 addrspace(1)* %addr2, align 8

  %add1 = add i64 %load2, %load1

  %addr3 = getelementptr inbounds i64, i64 addrspace(1)* %addr1, i64 536870656
  %load3 = load i64, i64 addrspace(1)* %addr3, align 8

  %add2 = add i64 %load3, %add1

  %addr4 = getelementptr inbounds i64, i64 addrspace(1)* %addr1, i64 536870912
  %load4 = load i64, i64 addrspace(1)* %addr4, align 8
  %add4 = add i64 %load4, %add2

  store i64 %add4, i64 addrspace(1)* %saddr, align 8
  ret void
}

; TODO: Support load4 as anchor instruction.
define amdgpu_kernel void @p32Offset64(i8 addrspace(1)*  %buffer) {
; GCN-LABEL: p32Offset64:
; GFX8:    flat_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}]
;
; GFX9:    global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX9:    global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off offset:2048
; GFX9:    global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off offset:3072
; GFX9:    global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off{{$}}
;
; GFX10:   global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off offset:-2048
; GFX10:   global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dword {{v[0-9]+}}, v[{{[0-9]+:[0-9]+}}], off offset:1024
entry:
  %call = tail call i64 @_Z13get_global_idj(i32 0)
  %conv = and i64 %call, 255
  %a0 = shl i64 %call, 7
  %idx.ext11 = and i64 %a0, 4294934528
  %add.ptr12 = getelementptr inbounds i8, i8 addrspace(1)* %buffer, i64 %idx.ext11
  %saddr = bitcast i8 addrspace(1)* %add.ptr12 to i32 addrspace(1)*

  %addr1 = getelementptr inbounds i32, i32 addrspace(1)* %saddr, i64 %conv
  %load1 = load i32, i32 addrspace(1)* %addr1, align 8

  %addr2 = getelementptr inbounds i32, i32 addrspace(1)* %addr1, i64 536870400
  %load2 = load i32, i32 addrspace(1)* %addr2, align 8

  %add1 = add i32 %load2, %load1

  %addr3 = getelementptr inbounds i32, i32 addrspace(1)* %addr1, i64 536870656
  %load3 = load i32, i32 addrspace(1)* %addr3, align 8

  %add2 = add i32 %load3, %add1

  %addr4 = getelementptr inbounds i32, i32 addrspace(1)* %addr1, i64 536870912
  %load4 = load i32, i32 addrspace(1)* %addr4, align 8
  %add4 = add i32 %load4, %add2

  store i32 %add4, i32 addrspace(1)* %saddr, align 8
  ret void
}

define amdgpu_kernel void @DiffBase(i8 addrspace(1)* %buffer1,
; GCN-LABEL: DiffBase:
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
;
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:2048
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:-4096
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:2048
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:2048
;
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:-2048
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:-2048
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
                                    i8 addrspace(1)* %buffer2) {
entry:
  %call = tail call i64 @_Z13get_global_idj(i32 0)
  %conv = and i64 %call, 255
  %a0 = shl i64 %call, 7
  %idx.ext11 = and i64 %a0, 4294934528
  %add.ptr12 = getelementptr inbounds i8, i8 addrspace(1)* %buffer1, i64 %idx.ext11
  %saddr = bitcast i8 addrspace(1)* %add.ptr12 to i64 addrspace(1)*

  %add.ptr2 = getelementptr inbounds i8, i8 addrspace(1)* %buffer2, i64 %idx.ext11
  %saddr2 = bitcast i8 addrspace(1)* %add.ptr2 to i64 addrspace(1)*

  %addr1 = getelementptr inbounds i64, i64 addrspace(1)* %saddr, i64 512
  %load1 = load i64, i64 addrspace(1)* %addr1, align 8
  %add.ptr8.3 = getelementptr inbounds i64, i64 addrspace(1)* %saddr, i64 768
  %load2 = load i64, i64 addrspace(1)* %add.ptr8.3, align 8
  %add1 = add i64 %load2, %load1
  %add.ptr8.4 = getelementptr inbounds i64, i64 addrspace(1)* %saddr, i64 1024
  %load3 = load i64, i64 addrspace(1)* %add.ptr8.4, align 8
  %add2 = add i64 %load3, %add1

  %add.ptr8.5 = getelementptr inbounds i64, i64 addrspace(1)* %saddr2, i64 1280
  %load4 = load i64, i64 addrspace(1)* %add.ptr8.5, align 8

  %add.ptr8.6 = getelementptr inbounds i64, i64 addrspace(1)* %saddr2, i64 1536
  %load5 = load i64, i64 addrspace(1)* %add.ptr8.6, align 8
  %add3 = add i64 %load5, %load4

  %add.ptr8.7 = getelementptr inbounds i64, i64 addrspace(1)* %saddr2, i64 1792
  %load6 = load i64, i64 addrspace(1)* %add.ptr8.7, align 8
  %add4 = add i64 %load6, %add3

  %add5 = add i64 %add2, %add4

  store i64 %add5, i64 addrspace(1)* %saddr, align 8
  ret void
}

define amdgpu_kernel void @ReverseOrder(i8 addrspace(1)* %buffer) {
; GCN-LABEL: ReverseOrder:
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
;
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:2048
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:2048
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:2048
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:2048
;
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
entry:
  %call = tail call i64 @_Z13get_global_idj(i32 0)
  %conv = and i64 %call, 255
  %a0 = shl i64 %call, 7
  %idx.ext11 = and i64 %a0, 4294934528
  %add.ptr12 = getelementptr inbounds i8, i8 addrspace(1)* %buffer, i64 %idx.ext11
  %saddr = bitcast i8 addrspace(1)* %add.ptr12 to i64 addrspace(1)*

  %addr1 = getelementptr inbounds i64, i64 addrspace(1)* %saddr, i64 %conv
  %load1 = load i64, i64 addrspace(1)* %addr1, align 8

  %add.ptr8.7 = getelementptr inbounds i64, i64 addrspace(1)* %addr1, i64 1792
  %load8 = load i64, i64 addrspace(1)* %add.ptr8.7, align 8
  %add7 = add i64 %load8, %load1

  %add.ptr8.6 = getelementptr inbounds i64, i64 addrspace(1)* %addr1, i64 1536
  %load7 = load i64, i64 addrspace(1)* %add.ptr8.6, align 8
  %add6 = add i64 %load7, %add7

  %add.ptr8.5 = getelementptr inbounds i64, i64 addrspace(1)* %addr1, i64 1280
  %load6 = load i64, i64 addrspace(1)* %add.ptr8.5, align 8
  %add5 = add i64 %load6, %add6

  %add.ptr8.4 = getelementptr inbounds i64, i64 addrspace(1)* %addr1, i64 1024
  %load5 = load i64, i64 addrspace(1)* %add.ptr8.4, align 8
  %add4 = add i64 %load5, %add5

  %add.ptr8.3 = getelementptr inbounds i64, i64 addrspace(1)* %addr1, i64 768
  %load4 = load i64, i64 addrspace(1)* %add.ptr8.3, align 8
  %add3 = add i64 %load4, %add4

  %add.ptr8.2 = getelementptr inbounds i64, i64 addrspace(1)* %addr1, i64 512
  %load3 = load i64, i64 addrspace(1)* %add.ptr8.2, align 8
  %add2 = add i64 %load3, %add3

  %addr2 = getelementptr inbounds i64, i64 addrspace(1)* %addr1, i64 256
  %load2 = load i64, i64 addrspace(1)* %addr2, align 8
  %add1 = add i64 %load2, %add2

  store i64 %add1, i64 addrspace(1)* %saddr, align 8
  ret void
}

define hidden amdgpu_kernel void @negativeoffset(i8 addrspace(1)* nocapture %buffer) {
; GCN-LABEL: negativeoffset:
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
; GFX8:    flat_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}]
;
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off offset:-2048
; GFX9:    global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
;
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
; GFX10:   global_load_dwordx2 v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], off{{$}}
entry:
  %call = tail call i64 @_Z13get_global_idj(i32 0) #2
  %conv = and i64 %call, 255
  %0 = shl i64 %call, 7
  %idx.ext11 = and i64 %0, 4294934528
  %add.ptr12 = getelementptr inbounds i8, i8 addrspace(1)* %buffer, i64 %idx.ext11
  %buffer_head = bitcast i8 addrspace(1)* %add.ptr12 to i64 addrspace(1)*

  %buffer_wave = getelementptr inbounds i64, i64 addrspace(1)* %buffer_head, i64 %conv

  %addr1 = getelementptr inbounds i64, i64 addrspace(1)* %buffer_wave, i64 -536870656
  %load1 = load i64, i64 addrspace(1)* %addr1, align 8

  %addr2 = getelementptr inbounds i64, i64 addrspace(1)* %buffer_wave, i64 -536870912
  %load2 = load i64, i64 addrspace(1)* %addr2, align 8


  %add = add i64 %load2, %load1

  store i64 %add, i64 addrspace(1)* %buffer_head, align 8
  ret void
}
