; RUN: llc -mtriple=hexagon-unknown-elf -mcpu=hexagonv55 -hexagon-initial-cfg-cleanup=0 < %s | FileCheck %s

; Test that the pipeliner correctly generates the operands in the
; epilog.

; CHECK: loop0
; CHECK: r{{[0-9]+}} = sfsub([[REG0:r([0-9]+)]],[[REG1:r([0-9]+)]])
; CHECK: endloop0
; CHECK: r{{[0-9]+}} = sfsub([[REG0]],[[REG1]])
; CHECK: r{{[0-9]+}} = sfsub([[REG0]],r{{[0-9]+}})

define dso_local void @test(i32 %m) local_unnamed_addr #0 {
entry:
  %div = sdiv i32 %m, 2
  %sub = add nsw i32 %div, -1
  br label %for.body.prol

for.body.prol:
  %i.0106.prol = phi i32 [ undef, %for.body.prol ], [ %sub, %entry ]
  %sr.prol = phi float [ %0, %for.body.prol ], [ undef, %entry ]
  %sr109.prol = phi float [ %sr.prol, %for.body.prol ], [ undef, %entry ]
  %prol.iter = phi i32 [ %prol.iter.sub, %for.body.prol ], [ undef, %entry ]
  %0 = load float, ptr undef, align 4
  %sub7.prol = fsub contract float %sr109.prol, %0
  store float %sub7.prol, ptr null, align 4
  %prol.iter.sub = add i32 %prol.iter, -1
  %prol.iter.cmp = icmp eq i32 %prol.iter.sub, 0
  br i1 %prol.iter.cmp, label %for.body.prol.loopexit, label %for.body.prol

for.body.prol.loopexit:
  unreachable
}

