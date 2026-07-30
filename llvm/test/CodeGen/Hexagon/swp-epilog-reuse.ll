; RUN: llc -fp-contract=fast -O3 -mtriple=hexagon -mcpu=hexagonv5 < %s
; REQUIRES: asserts

; Test that the pipeliner doesn't ICE due because the PHI generation
; code in the epilog does not attempt to reuse an existing PHI.

define void @test(ptr noalias %srcImg, i32 %width, ptr noalias %dstImg) {
entry.split:
  %shr = lshr i32 %width, 1
  %incdec.ptr253 = getelementptr inbounds float, ptr %dstImg, i32 2
  br i1 undef, label %for.body, label %for.end

for.body:
  %dst.21518.reg2mem.0 = phi ptr [ null, %while.end712 ], [ %incdec.ptr253, %entry.split ]
  %dstEnd.01519 = phi ptr [ %add.ptr725, %while.end712 ], [ undef, %entry.split ]
  %add.ptr367 = getelementptr inbounds float, ptr %srcImg, i32 undef
  %dst.31487 = getelementptr inbounds float, ptr %dst.21518.reg2mem.0, i32 1
  br i1 undef, label %while.body661.preheader, label %while.end712

while.body661.preheader:
  %scevgep1941 = getelementptr float, ptr %add.ptr367, i32 1
  br label %while.body661.ur

while.body661.ur:
  %lsr.iv1942 = phi ptr [ %scevgep1941, %while.body661.preheader ], [ undef, %while.body661.ur ]
  %col1.31508.reg2mem.0.ur = phi float [ %col3.31506.reg2mem.0.ur, %while.body661.ur ], [ undef, %while.body661.preheader ]
  %col4.31507.reg2mem.0.ur = phi float [ %add710.ur, %while.body661.ur ], [ 0.000000e+00, %while.body661.preheader ]
  %col3.31506.reg2mem.0.ur = phi float [ %add689.ur, %while.body661.ur ], [ undef, %while.body661.preheader ]
  %dst.41511.ur = phi ptr [ %incdec.ptr674.ur, %while.body661.ur ], [ %dst.31487, %while.body661.preheader ]
  %mul662.ur = fmul float %col1.31508.reg2mem.0.ur, 4.000000e+00
  %add663.ur = fadd float undef, %mul662.ur
  %add665.ur = fadd float %add663.ur, undef
  %add667.ur = fadd float undef, %add665.ur
  %add669.ur = fadd float undef, %add667.ur
  %add670.ur = fadd float %col4.31507.reg2mem.0.ur, %add669.ur
  %conv673.ur = fmul float %add670.ur, 3.906250e-03
  %incdec.ptr674.ur = getelementptr inbounds float, ptr %dst.41511.ur, i32 1
  store float %conv673.ur, ptr %dst.41511.ur, align 4
  %scevgep1959 = getelementptr float, ptr %lsr.iv1942, i32 -1
  %0 = load float, ptr %scevgep1959, align 4
  %mul680.ur = fmul float %0, 4.000000e+00
  %add681.ur = fadd float undef, %mul680.ur
  %add684.ur = fadd float undef, %add681.ur
  %add687.ur = fadd float undef, %add684.ur
  %add689.ur = fadd float undef, %add687.ur
  %add699.ur = fadd float undef, undef
  %add703.ur = fadd float undef, %add699.ur
  %add707.ur = fadd float undef, %add703.ur
  %add710.ur = fadd float undef, %add707.ur
  %cmp660.ur = icmp ult ptr %incdec.ptr674.ur, %dstEnd.01519
  br i1 %cmp660.ur, label %while.body661.ur, label %while.end712

while.end712:
  %dst.4.lcssa.reg2mem.0 = phi ptr [ %dst.31487, %for.body ], [ undef, %while.body661.ur ]
  %conv721 = fpext float undef to double
  %mul722 = fmul double %conv721, 0x3F7111112119E8FB
  %conv723 = fptrunc double %mul722 to float
  store float %conv723, ptr %dst.4.lcssa.reg2mem.0, align 4
  %add.ptr725 = getelementptr inbounds float, ptr %dstEnd.01519, i32 %shr
  %cmp259 = icmp ult i32 undef, undef
  br i1 %cmp259, label %for.body, label %for.end

for.end:
  ret void
}
