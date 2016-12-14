; RUN: opt -mtriple=r600-- -amdgpu-printf-runtime-binding -mcpu=r600 -S < %s | FileCheck --check-prefix=FUNC --check-prefix=R600 %s
; RUN: opt -mtriple=amdgcn-- -amdgpu-printf-runtime-binding -mcpu=fiji -S < %s | FileCheck --check-prefix=FUNC --check-prefix=GCN %s
; RUN: opt -mtriple=amdgcn--amdhsa -amdgpu-printf-runtime-binding -mcpu=fiji -S < %s | FileCheck --check-prefix=FUNC --check-prefix=GCN %s

; FUNC-LABEL: @test_kernel(
; R600-LABEL: entry
; R600-NOT: call i8 addrspace(1)* @__printf_alloc
; R600: call i32 (i8 addrspace(2)*, ...) @printf(i8 addrspace(2)* getelementptr inbounds ([6 x i8], [6 x i8] addrspace(2)* @.str, i32 0, i32 0), i8* %arraydecay, i32 %3)
; GCN-LABEL: entry
; GCN: call i8 addrspace(1)* @__printf_alloc
; GCN-LABEL: entry.split
; GCN: icmp ne i8 addrspace(1)* %printf_alloc_fn, null
; GCN: %PrintBuffID = getelementptr i8, i8 addrspace(1)* %printf_alloc_fn, i32 0
; GCN: %PrintBuffIdCast = bitcast i8 addrspace(1)* %PrintBuffID to i32 addrspace(1)*
; GCN: store i32 1, i32 addrspace(1)* %PrintBuffIdCast
; GCN: %PrintBuffGep = getelementptr i8, i8 addrspace(1)* %printf_alloc_fn, i32 4
; GCN: %PrintArgPtr = ptrtoint i8* %arraydecay to i64
; GCN: %PrintBuffPtrCast = bitcast i8 addrspace(1)* %PrintBuffGep to i64 addrspace(1)*
; GCN: store i64 %PrintArgPtr, i64 addrspace(1)* %PrintBuffPtrCast
; GCN: %PrintBuffNextPtr = getelementptr i8, i8 addrspace(1)* %PrintBuffGep, i32 8
; GCN: %PrintBuffPtrCast1 = bitcast i8 addrspace(1)* %PrintBuffNextPtr to i32 addrspace(1)*
; GCN: store i32 %3, i32 addrspace(1)* %PrintBuffPtrCast1

@test_kernel.str = private unnamed_addr constant [9 x i8] c"globalid\00", align 1
@.str = private unnamed_addr addrspace(2) constant [6 x i8] c"%s:%d\00", align 1

define amdgpu_kernel void @test_kernel(i32 addrspace(1)* %in, i32 addrspace(1)* %out) {
entry:
  %in.addr = alloca i32 addrspace(1)*, align 4
  %out.addr = alloca i32 addrspace(1)*, align 4
  %n = alloca i32, align 4
  %str = alloca [9 x i8], align 1
  store i32 addrspace(1)* %in, i32 addrspace(1)** %in.addr, align 4
  store i32 addrspace(1)* %out, i32 addrspace(1)** %out.addr, align 4
  %0 = bitcast i32* %n to i8*
  %call = call i64 @_Z13get_global_idj(i32 0) #5
  %conv = trunc i64 %call to i32
  store i32 %conv, i32* %n, align 4
  %1 = bitcast [9 x i8]* %str to i8*
  %2 = bitcast [9 x i8]* %str to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %2, i8* getelementptr inbounds ([9 x i8], [9 x i8]* @test_kernel.str, i32 0, i32 0), i64 9, i32 1, i1 false)
  %arraydecay = getelementptr inbounds [9 x i8], [9 x i8]* %str, i32 0, i32 0
  %3 = load i32, i32* %n, align 4
  %call1 = call i32 (i8 addrspace(2)*, ...) @printf(i8 addrspace(2)* getelementptr inbounds ([6 x i8], [6 x i8] addrspace(2)* @.str, i32 0, i32 0), i8* %arraydecay, i32 %3)
  %4 = load i32, i32* %n, align 4
  %idxprom = sext i32 %4 to i64
  %5 = load i32 addrspace(1)*, i32 addrspace(1)** %in.addr, align 4
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %5, i64 %idxprom
  %6 = load i32, i32 addrspace(1)* %arrayidx, align 4
  %7 = load i32, i32* %n, align 4
  %idxprom2 = sext i32 %7 to i64
  %8 = load i32 addrspace(1)*, i32 addrspace(1)** %out.addr, align 4
  %arrayidx3 = getelementptr inbounds i32, i32 addrspace(1)* %8, i64 %idxprom2
  store i32 %6, i32 addrspace(1)* %arrayidx3, align 4
  %9 = bitcast [9 x i8]* %str to i8*
  %10 = bitcast i32* %n to i8*
  ret void
}

; Function Attrs: nounwind readnone
declare i64 @_Z13get_global_idj(i32) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #1

declare i32 @printf(i8 addrspace(2)*, ...) #3
