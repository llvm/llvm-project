; RUN: llc -march=amdgcn -mcpu=gfx900 -O0 -verify-machineinstrs < %s -debug-only=isel 2>&1 | FileCheck --check-prefixes=GCN,GCN-DEFAULT %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -O0 -verify-machineinstrs < %s -debug-only=isel -dag-dump-verbose 2>&1 | FileCheck --check-prefixes=GCN,GCN-VERBOSE %s

; REQUIRES: asserts

; GCN-LABEL: === test_sdag_dump
; GCN: Initial selection DAG: %bb.0 'test_sdag_dump:entry'
; GCN: SelectionDAG has 10 nodes:

; GCN-DEFAULT:  t0: ch,glue = EntryToken
; GCN-DEFAULT:  t2: f32,ch = CopyFromReg t0, Register:f32 %0
; GCN-DEFAULT:      t5: f32 = fadd t2, t2
; GCN-DEFAULT:      t4: f32,ch = CopyFromReg # D:1 t0, Register:f32 %1
; GCN-DEFAULT:    t6: f32 = fadd # D:1 t5, t4
; GCN-DEFAULT:  t8: ch,glue = CopyToReg # D:1 t0, Register:f32 $vgpr0, t6
; GCN-DEFAULT:  t9: ch = RETURN_TO_EPILOG # D:1 t8, Register:f32 $vgpr0, t8:1

; GCN-VERBOSE:  t0: ch,glue = EntryToken # D:0
; GCN-VERBOSE:  t2: f32,ch = CopyFromReg [ORD=1] # D:0 t0, Register:f32 %0 # D:0
; GCN-VERBOSE:      t5: f32 = fadd [ORD=2] # D:0 t2, t2
; GCN-VERBOSE:      t4: f32,ch = CopyFromReg [ORD=1] # D:1 t0, Register:f32 %1 # D:0
; GCN-VERBOSE:    t6: f32 = fadd [ORD=3] # D:1 t5, t4
; GCN-VERBOSE:  t8: ch,glue = CopyToReg [ORD=4] # D:1 t0, Register:f32 $vgpr0 # D:0, t6
; GCN-VERBOSE:  t9: ch = RETURN_TO_EPILOG [ORD=4] # D:1 t8, Register:f32 $vgpr0 # D:0, t8:1

define amdgpu_ps float @test_sdag_dump(float inreg %scalar, float %vector)  {
entry:
  %sadd = fadd float %scalar, %scalar
  %ret = fadd float %sadd, %vector
  ret float %ret
}

declare i32 @llvm.amdgcn.workitem.id.x()
