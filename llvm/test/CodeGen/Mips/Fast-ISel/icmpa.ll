; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32r2 \
; RUN:     < %s | FileCheck %s
; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32 \
; RUN:     < %s | FileCheck %s

@c = global i32 4, align 4
@d = global i32 9, align 4
@uc = global i32 4, align 4
@ud = global i32 9, align 4
@b1 = common global i32 0, align 4

; Function Attrs: nounwind
define void @eq()  {
entry:
; CHECK-LABEL:  .ent  eq

  %0 = load i32, ptr @c, align 4
  %1 = load i32, ptr @d, align 4
  %cmp = icmp eq i32 %0, %1
  %conv = zext i1 %cmp to i32
; CHECK-DAG:  lw	$[[REG_D_GOT:[0-9+]]], %got(d)(${{[0-9]+}})
; CHECK-DAG:  lw	$[[REG_C_GOT:[0-9+]]], %got(c)(${{[0-9]+}})
; CHECK-DAG:  lw	$[[REG_D:[0-9]+]], 0($[[REG_D_GOT]])
; CHECK-DAG:  lw	$[[REG_C:[0-9]+]], 0($[[REG_C_GOT]])
; CHECK:  xor  $[[REG1:[0-9]+]], $[[REG_C]], $[[REG_D]]
; CHECK:  sltiu  $[[REG2:[0-9]+]], $[[REG1]], 1
; FIXME: This instruction is redundant. The sltiu can only produce 0 and 1.
; CHECK:  andi  ${{[0-9]+}}, $[[REG2]], 1

  store i32 %conv, ptr @b1, align 4
  ret void
}

; Function Attrs: nounwind
define void @ne()  {
entry:
; CHECK-LABEL:  .ent  ne
  %0 = load i32, ptr @c, align 4
  %1 = load i32, ptr @d, align 4
  %cmp = icmp ne i32 %0, %1
  %conv = zext i1 %cmp to i32
; CHECK-DAG:  lw	$[[REG_D_GOT:[0-9+]]], %got(d)(${{[0-9]+}})
; CHECK-DAG:  lw	$[[REG_C_GOT:[0-9+]]], %got(c)(${{[0-9]+}})
; CHECK-DAG:  lw	$[[REG_D:[0-9]+]], 0($[[REG_D_GOT]])
; CHECK-DAG:  lw	$[[REG_C:[0-9]+]], 0($[[REG_C_GOT]])
; CHECK:  xor  $[[REG1:[0-9]+]], $[[REG_C]], $[[REG_D]]
; CHECK:  sltu  $[[REG2:[0-9]+]], $zero, $[[REG1]]
; FIXME: This instruction is redundant. The sltu can only produce 0 and 1.
; CHECK:  andi  ${{[0-9]+}}, $[[REG2]], 1

  store i32 %conv, ptr @b1, align 4
  ret void
}

; Function Attrs: nounwind
define void @ugt()  {
entry:
; CHECK-LABEL:  .ent  ugt
  %0 = load i32, ptr @uc, align 4
  %1 = load i32, ptr @ud, align 4
  %cmp = icmp ugt i32 %0, %1
  %conv = zext i1 %cmp to i32
; CHECK:  lw	$[[REG_UC_GOT:[0-9+]]], %got(uc)(${{[0-9]+}})
; CHECK:  lw	$[[REG_UC:[0-9]+]], 0($[[REG_UC_GOT]])
; CHECK:  lw	$[[REG_UD_GOT:[0-9+]]], %got(ud)(${{[0-9]+}})
; CHECK:  lw	$[[REG_UD:[0-9]+]], 0($[[REG_UD_GOT]])
; CHECK:  sltu  $[[REG1:[0-9]+]], $[[REG_UD]], $[[REG_UC]]
; FIXME: This instruction is redundant. The sltu can only produce 0 and 1.
; CHECK:  andi  ${{[0-9]+}}, $[[REG1]], 1

  store i32 %conv, ptr @b1, align 4
  ret void
}

; Function Attrs: nounwind
define void @ult()  {
entry:
; CHECK-LABEL:  .ent  ult
  %0 = load i32, ptr @uc, align 4
  %1 = load i32, ptr @ud, align 4
  %cmp = icmp ult i32 %0, %1
  %conv = zext i1 %cmp to i32
; CHECK-DAG:  lw	$[[REG_UD_GOT:[0-9+]]], %got(ud)(${{[0-9]+}})
; CHECK-DAG:  lw	$[[REG_UC_GOT:[0-9+]]], %got(uc)(${{[0-9]+}})
; CHECK-DAG:  lw	$[[REG_UD:[0-9]+]], 0($[[REG_UD_GOT]])
; CHECK-DAG:  lw	$[[REG_UC:[0-9]+]], 0($[[REG_UC_GOT]])
; CHECK:  sltu  $[[REG1:[0-9]+]], $[[REG_UC]], $[[REG_UD]]
; FIXME: This instruction is redundant. The sltu can only produce 0 and 1.
; CHECK:  andi  ${{[0-9]+}}, $[[REG1]], 1
  store i32 %conv, ptr @b1, align 4
  ret void
}

; Function Attrs: nounwind
define void @uge()  {
entry:
; CHECK-LABEL:  .ent  uge
  %0 = load i32, ptr @uc, align 4
  %1 = load i32, ptr @ud, align 4
  %cmp = icmp uge i32 %0, %1
  %conv = zext i1 %cmp to i32
; CHECK-DAG:  lw	$[[REG_UD_GOT:[0-9+]]], %got(ud)(${{[0-9]+}})
; CHECK-DAG:  lw	$[[REG_UC_GOT:[0-9+]]], %got(uc)(${{[0-9]+}})
; CHECK-DAG:  lw	$[[REG_UD:[0-9]+]], 0($[[REG_UD_GOT]])
; CHECK-DAG:  lw	$[[REG_UC:[0-9]+]], 0($[[REG_UC_GOT]])
; CHECK:  sltu  $[[REG1:[0-9]+]], $[[REG_UC]], $[[REG_UD]]
; CHECK:  xori  $[[REG2:[0-9]+]], $[[REG1]], 1
; FIXME: This instruction is redundant. The sltu can only produce 0 and 1.
; CHECK:  andi  ${{[0-9]+}}, $[[REG2]], 1
  store i32 %conv, ptr @b1, align 4
  ret void
}

; Function Attrs: nounwind
define void @ule()  {
entry:
; CHECK-LABEL:  .ent  ule
  %0 = load i32, ptr @uc, align 4
  %1 = load i32, ptr @ud, align 4
  %cmp = icmp ule i32 %0, %1
  %conv = zext i1 %cmp to i32
; CHECK:  lw	$[[REG_UC_GOT:[0-9+]]], %got(uc)(${{[0-9]+}})
; CHECK:  lw	$[[REG_UC:[0-9]+]], 0($[[REG_UC_GOT]])
; CHECK:  lw	$[[REG_UD_GOT:[0-9+]]], %got(ud)(${{[0-9]+}})
; CHECK:  lw	$[[REG_UD:[0-9]+]], 0($[[REG_UD_GOT]])
; CHECK:  sltu  $[[REG1:[0-9]+]], $[[REG_UD]], $[[REG_UC]]
; CHECK:  xori  $[[REG2:[0-9]+]], $[[REG1]], 1
; FIXME: This instruction is redundant. The sltu can only produce 0 and 1.
; CHECK:  andi  ${{[0-9]+}}, $[[REG2]], 1
  store i32 %conv, ptr @b1, align 4
  ret void
}

; Function Attrs: nounwind
define void @sgt()  {
entry:
; CHECK-LABEL:  .ent sgt
  %0 = load i32, ptr @c, align 4
  %1 = load i32, ptr @d, align 4
  %cmp = icmp sgt i32 %0, %1
  %conv = zext i1 %cmp to i32
; CHECK:  lw	$[[REG_C_GOT:[0-9+]]], %got(c)(${{[0-9]+}})
; CHECK:  lw	$[[REG_C:[0-9]+]], 0($[[REG_C_GOT]])
; CHECK:  lw	$[[REG_D_GOT:[0-9+]]], %got(d)(${{[0-9]+}})
; CHECK:  lw	$[[REG_D:[0-9]+]], 0($[[REG_D_GOT]])
; CHECK:  slt  $[[REG1:[0-9]+]], $[[REG_D]], $[[REG_C]]
; FIXME: This instruction is redundant. The slt can only produce 0 and 1.
; CHECK:  andi  ${{[0-9]+}}, $[[REG1]], 1
  store i32 %conv, ptr @b1, align 4
  ret void
}

; Function Attrs: nounwind
define void @slt()  {
entry:
; CHECK-LABEL:  .ent slt
  %0 = load i32, ptr @c, align 4
  %1 = load i32, ptr @d, align 4
  %cmp = icmp slt i32 %0, %1
  %conv = zext i1 %cmp to i32
; CHECK-DAG:  lw	$[[REG_D_GOT:[0-9+]]], %got(d)(${{[0-9]+}})
; CHECK-DAG:  lw	$[[REG_C_GOT:[0-9+]]], %got(c)(${{[0-9]+}})
; CHECK-DAG:  lw	$[[REG_D:[0-9]+]], 0($[[REG_D_GOT]])
; CHECK-DAG:  lw	$[[REG_C:[0-9]+]], 0($[[REG_C_GOT]])
; CHECK:  slt  $[[REG1:[0-9]+]], $[[REG_C]], $[[REG_D]]
; FIXME: This instruction is redundant. The slt can only produce 0 and 1.
; CHECK:  andi  ${{[0-9]+}}, $[[REG1]], 1
  store i32 %conv, ptr @b1, align 4
  ret void
}

; Function Attrs: nounwind
define void @sge()  {
entry:
; CHECK-LABEL:  .ent sge
  %0 = load i32, ptr @c, align 4
  %1 = load i32, ptr @d, align 4
  %cmp = icmp sge i32 %0, %1
  %conv = zext i1 %cmp to i32
  store i32 %conv, ptr @b1, align 4
; CHECK-DAG:  lw	$[[REG_D_GOT:[0-9+]]], %got(d)(${{[0-9]+}})
; CHECK-DAG:  lw	$[[REG_C_GOT:[0-9+]]], %got(c)(${{[0-9]+}})
; CHECK-DAG:  lw	$[[REG_D:[0-9]+]], 0($[[REG_D_GOT]])
; CHECK-DAG:  lw	$[[REG_C:[0-9]+]], 0($[[REG_C_GOT]])
; CHECK:  slt  $[[REG1:[0-9]+]], $[[REG_C]], $[[REG_D]]
; CHECK:  xori  $[[REG2:[0-9]+]], $[[REG1]], 1
; FIXME: This instruction is redundant. The slt can only produce 0 and 1.
; CHECK:  andi  ${{[0-9]+}}, $[[REG2]], 1
  ret void
}

; Function Attrs: nounwind
define void @sle()  {
entry:
; CHECK-LABEL:  .ent sle
  %0 = load i32, ptr @c, align 4
  %1 = load i32, ptr @d, align 4
  %cmp = icmp sle i32 %0, %1
  %conv = zext i1 %cmp to i32
; CHECK:  lw	$[[REG_C_GOT:[0-9+]]], %got(c)(${{[0-9]+}})
; CHECK:  lw	$[[REG_C:[0-9]+]], 0($[[REG_C_GOT]])
; CHECK:  lw	$[[REG_D_GOT:[0-9+]]], %got(d)(${{[0-9]+}})
; CHECK:  lw	$[[REG_D:[0-9]+]], 0($[[REG_D_GOT]])
; CHECK:        slt     $[[REG1:[0-9]+]], $[[REG_D]], $[[REG_C]]
; CHECK:        xori    $[[REG2:[0-9]+]], $[[REG1]], 1
; FIXME: This instruction is redundant. The slt can only produce 0 and 1.
; CHECK:        andi    ${{[0-9]+}}, $[[REG2]], 1
  store i32 %conv, ptr @b1, align 4
  ret void
}
