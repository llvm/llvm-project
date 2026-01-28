; RUN: opt -mtriple amdgcn-- -mcpu=gfx900 -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck --check-prefixes=GCN,GFX9 %s
; RUN: opt -mtriple amdgcn-- -mcpu=gfx1010 -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck --check-prefixes=GCN,GFX10 %s

; GCN: UniformityInfo for function 'tidx_shift_6':
; GCN: DIVERGENT: %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
; GCN: {{^}} %shr = lshr i32 %tidx, 6
define i32 @tidx_shift_6() {
entry:
  %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
  %shr = lshr i32 %tidx, 6
  ret i32 %shr
}

; GCN: UniformityInfo for function 'tidx_shift_7':
; GCN: DIVERGENT: %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
; GCN: {{^}} %shr = lshr i32 %tidx, 7
define i32 @tidx_shift_7() {
entry:
  %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
  %shr = lshr i32 %tidx, 7
  ret i32 %shr
}

; GCN: UniformityInfo for function 'tidx_shift_5':
; GCN: DIVERGENT:  %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
; GFX9: DIVERGENT: %shr = lshr i32 %tidx, 5
; GFX10: {{^}} %shr = lshr i32 %tidx, 5
define i32 @tidx_shift_5() {
entry:
  %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
  %shr = lshr i32 %tidx, 5
  ret i32 %shr
}

; GCN: UniformityInfo for function 'tidx_shift_4':
; GCN: DIVERGENT: %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
; GCN: DIVERGENT: %shr = lshr i32 %tidx, 4
define i32 @tidx_shift_4() {
entry:
  %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
  %shr = lshr i32 %tidx, 4
  ret i32 %shr
}

; GCN: UniformityInfo for function 'tidx_and_63':
; GCN: DIVERGENT: %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
; GCN: DIVERGENT: %and = and i32 %tidx, 63
define i32 @tidx_and_63() {
entry:
  %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
  %and = and i32 %tidx, 63
  ret i32 %and
}

; GCN: UniformityInfo for function 'tidx_and_64':
; GCN: DIVERGENT: %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
; GCN: {{^}} %and = and i32 %tidx, 64
define i32 @tidx_and_64() {
entry:
  %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
  %and = and i32 %tidx, 64
  ret i32 %and
}

; GCN: UniformityInfo for function 'tidx_and_65':
; GCN: DIVERGENT: %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
; GCN: DIVERGENT: %and = and i32 %tidx, 65
define i32 @tidx_and_65() {
entry:
  %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
  %and = and i32 %tidx, 65
  ret i32 %and
}

; GCN: UniformityInfo for function 'tidx_and_320':
; GCN: DIVERGENT: %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
; GCN: {{^}} %and = and i32 %tidx, 320
define i32 @tidx_and_320() {
entry:
  %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
  %and = and i32 %tidx, 320
  ret i32 %and
}

; GCN: UniformityInfo for function 'tidx_and_31':
; GCN: DIVERGENT: %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
; GCN: DIVERGENT: %and = and i32 %tidx, 31
define i32 @tidx_and_31() {
entry:
  %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
  %and = and i32 %tidx, 31
  ret i32 %and
}

; GCN: UniformityInfo for function 'tidx_and_32':
; GCN: DIVERGENT: %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
; GFX9: DIVERGENT: %and = and i32 %tidx, 32
; GFX10: {{^}} %and = and i32 %tidx, 32
define i32 @tidx_and_32() {
entry:
  %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
  %and = and i32 %tidx, 32
  ret i32 %and
}

; GCN: UniformityInfo for function 'tidx_or_63':
; GCN: DIVERGENT: %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
; GCN: {{^}} %or = or i32 %tidx, 63
define i32 @tidx_or_63() {
entry:
  %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
  %or = or i32 %tidx, 63
  ret i32 %or
}

; GCN: UniformityInfo for function 'tidx_or_64':
; GCN: DIVERGENT: %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
; GCN: DIVERGENT: %or = or i32 %tidx, 64
define i32 @tidx_or_64() {
entry:
  %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
  %or = or i32 %tidx, 64
  ret i32 %or
}

; GCN: UniformityInfo for function 'tidx_or_65':
; GCN: DIVERGENT: %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
; GCN: DIVERGENT: %or = or i32 %tidx, 65
define i32 @tidx_or_65() {
entry:
  %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
  %or = or i32 %tidx, 65
  ret i32 %or
}

; GCN: UniformityInfo for function 'tidx_or_31':
; GCN: DIVERGENT: %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
; GFX9: DIVERGENT: %or = or i32 %tidx, 31
; GFX10: {{^}} %or = or i32 %tidx, 31
define i32 @tidx_or_31() {
entry:
  %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
  %or = or i32 %tidx, 31
  ret i32 %or
}

; GCN: UniformityInfo for function 'tidx_or_32':
; GCN: DIVERGENT: %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
; GCN: DIVERGENT: %or = or i32 %tidx, 32
define i32 @tidx_or_32() {
entry:
  %tidx = tail call i32 @llvm.amdgcn.workitem.id.x()
  %or = or i32 %tidx, 32
  ret i32 %or
}
