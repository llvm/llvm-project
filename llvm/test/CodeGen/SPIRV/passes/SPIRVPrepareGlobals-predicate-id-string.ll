; RUN: llc -O0 -mtriple=spirv64-amd-amdhsa %s -print-after-all -o - 2>&1 | FileCheck %s
; RUN: llc -O3 -mtriple=spirv64-amd-amdhsa %s -print-after-all -o - 2>&1 | FileCheck %s

; CHECK: *** IR Dump After SPIRV prepare global variables (prepare-globals) ***

declare void @llvm.amdgcn.s.monitor.sleep(i16 immarg) addrspace(4)

declare void @llvm.amdgcn.s.sleep(i32 immarg) addrspace(4)

declare i1 @_Z20__spirv_SpecConstantib(i32, i1) addrspace(4)

declare i16 @llvm.amdgcn.ashr.pk.i8.i32(i32, i32, i32) addrspace(4) #3

declare void @llvm.set.fpenv.i64(i64) addrspace(4) #4

declare void @llvm.amdgcn.s.sleep.var(i32) addrspace(4) #5

declare void @llvm.amdgcn.s.wait.event.export.ready() addrspace(4) #5

declare void @llvm.amdgcn.s.ttracedata.imm(i16 immarg) addrspace(4) #6

@p = external addrspace(1) global i32
@g = external addrspace(1) constant i32
; CHECK: @llvm.amdgcn.feature.predicate.ids = addrspace(1) constant [99 x i8] c"is.gfx1010 2\00is.gfx950 6\00is.gfx1101 3\00has.gfx11-insts 0\00is.gfx906 4\00is.gfx1201 1\00has.gfx12-insts 5\00"

define void @kernel() addrspace(4) {
entry:
  %x = load i32, ptr addrspace(1) @g
  %is.gfx950. = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !9
  br i1 %is.gfx950., label %cond.true, label %cond.false
cond.true:
  %0 = call addrspace(4) i16 @llvm.amdgcn.ashr.pk.i8.i32(i32 8, i32 8, i32 8)
  br label %cond.end
cond.false:
  call addrspace(4) void @llvm.set.fpenv.i64(i64 -1)
  br label %cond.end
cond.end:
  %is.gfx1201. = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !9
  br i1 %is.gfx1201., label %if.then, label %lor.lhs.false
lor.lhs.false:
  %has.gfx12-insts. = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !10
  br i1 %has.gfx12-insts., label %if.then, label %if.end
if.then:
  call addrspace(4) void @llvm.amdgcn.s.sleep.var(i32 %x)
  br label %if.end
if.end:
  %is.gfx906. = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !11
  br i1 %is.gfx906., label %if.else, label %if.then2
if.then2:
  call addrspace(4) void @llvm.amdgcn.s.wait.event.export.ready()
  br label %if.end6
if.else:
  %is.gfx1010. = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !12
  br i1 %is.gfx1010., label %if.then4, label %lor.lhs.false3
lor.lhs.false3:
  %is.gfx1101. = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !13
  br i1 %is.gfx1101., label %if.then4, label %if.end5
if.then4:
  call addrspace(4) void @llvm.amdgcn.s.ttracedata.imm(i16 1)
  br label %if.end5
if.end5:
  br label %if.end6
if.end6:
  br label %while.cond
while.cond:
  %is.gfx1101.7 = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !14
  br i1 %is.gfx1101.7, label %while.body, label %while.end
while.body:
  %4 = load i32, ptr addrspace(1) @p
  %add = add i32 %4, %x
  store i32 %add, ptr addrspace(1) @p
  br label %while.end
while.end:
  br label %do.body
do.body:
  %7 = load i32, ptr addrspace(1) @p
  %sub = sub i32 %7, %x
  store i32 %sub, ptr addrspace(1) @p
  br label %do.end
do.cond:
  %is.gfx1010.8 = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !14
  br i1 %is.gfx1010.8, label %do.body, label %do.end
do.end:
  br label %for.cond
for.cond:
  %is.gfx1201.9 = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !13
  br i1 %is.gfx1201.9, label %for.body, label %for.end
for.body:
  br label %for.end
for.inc:
  %9 = load i32, ptr addrspace(1) @p
  %inc = add i32 %9, 1
  store i32 %inc, ptr addrspace(1) @p
  br label %for.cond
for.end:
  %has.gfx11-insts. = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !10
  br i1 %has.gfx11-insts., label %if.then10, label %if.else11
if.then10:
  call addrspace(4) void @llvm.amdgcn.s.wait.event.export.ready()
  br label %if.end14
if.else11:
  %has.gfx10-insts. = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !18
  br i1 %has.gfx10-insts., label %if.then12, label %if.end13
if.then12:
  call addrspace(4) void @llvm.amdgcn.s.ttracedata.imm(i16 1)
  br label %if.end13
if.end13:
  br label %if.end14
if.end14:
  br label %do.body15
do.body15:
  %12 = load i32, ptr addrspace(1) @p
  %sub16 = sub i32 %12, %x
  store i32 %sub16, ptr addrspace(1) @p
  br label %do.end18
do.cond17:
  %has.gfx1250-insts. = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !20
  br i1 %has.gfx1250-insts., label %do.body15, label %do.end18
do.end18:
  br label %for.cond19
for.cond19:
  %has.gfx11-insts.20 = call addrspace(4) i1 @_Z20__spirv_SpecConstantib(i32 -1, i1 false), !llvm.amdgcn.feature.predicate !18
  br i1 %has.gfx11-insts.20, label %for.body21, label %for.end24
for.body21:
  br label %for.end24
for.inc22:
  %14 = load i32, ptr addrspace(1) @p
  %inc23 = add i32 %14, 1
  store i32 %inc23, ptr addrspace(1) @p
  br label %for.cond19
for.end24:
  ret void
}

!9 = !{!"is.gfx950"}
!10 = !{!"is.gfx1201"}
!11 = !{!"has.gfx12-insts"}
!12 = !{!"is.gfx906"}
!13 = !{!"is.gfx1010"}
!14 = !{!"is.gfx1101"}
!18 = !{!"has.gfx11-insts"}
!20 = !{!"has.gfx1250-insts"}