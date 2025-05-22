; RUN: llc -mtriple=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32r2 \
; RUN:     < %s | FileCheck %s
; RUN: llc -mtriple=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32 \
; RUN:     < %s | FileCheck %s
; RUN: llc -mtriple=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32r2 \
; RUN:     < %s | FileCheck %s -check-prefix=mips32r2
; RUN: llc -mtriple=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32 \
; RUN:     < %s | FileCheck %s -check-prefix=mips32

@b2 = global i8 0, align 1
@b1 = global i8 1, align 1
@uc1 = global i8 0, align 1
@uc2 = global i8 -1, align 1
@sc1 = global i8 -128, align 1
@sc2 = global i8 127, align 1
@ss1 = global i16 -32768, align 2
@ss2 = global i16 32767, align 2
@us1 = global i16 0, align 2
@us2 = global i16 -1, align 2
@ssi = global i16 0, align 2
@ssj = global i16 0, align 2
@i = global i32 0, align 4
@j = global i32 0, align 4
@.str = private unnamed_addr constant [4 x i8] c"%i\0A\00", align 1
@.str1 = private unnamed_addr constant [7 x i8] c"%i %i\0A\00", align 1

; Function Attrs: nounwind
define void @_Z3b_iv()  {
entry:
; CHECK-LABEL:   .ent  _Z3b_iv
  %0 = load i8, ptr @b1, align 1
  %tobool = trunc i8 %0 to i1
  %frombool = zext i1 %tobool to i8
  store i8 %frombool, ptr @b2, align 1
  %1 = load i8, ptr @b2, align 1
  %tobool1 = trunc i8 %1 to i1
  %conv = zext i1 %tobool1 to i32
  store i32 %conv, ptr @i, align 4
; CHECK:  lbu  $[[REG1:[0-9]+]], 0(${{[0-9]+}})
; CHECK:  andi  $[[REG2:[0-9]+]], $[[REG1]], 1
; CHECK:  sb  $[[REG2]], 0(${{[0-9]+}})



  ret void
; CHECK:   .end  _Z3b_iv
}

; Function Attrs: nounwind
define void @_Z4uc_iv()  {
entry:
; CHECK-LABEL:  .ent  _Z4uc_iv

  %0 = load i8, ptr @uc1, align 1
  %conv = zext i8 %0 to i32
  store i32 %conv, ptr @i, align 4
  %1 = load i8, ptr @uc2, align 1
  %conv1 = zext i8 %1 to i32
; CHECK:   lbu  $[[REG1:[0-9]+]], 0(${{[0-9]+}})
; CHECK:  andi  ${{[0-9]+}}, $[[REG1]], 255

  store i32 %conv1, ptr @j, align 4
  ret void
; CHECK:  .end  _Z4uc_iv

}

; Function Attrs: nounwind
define void @_Z4sc_iv()  {
entry:
; mips32r2-LABEL:  .ent  _Z4sc_iv
; mips32-LABEL:  .ent  _Z4sc_iv

  %0 = load i8, ptr @sc1, align 1
  %conv = sext i8 %0 to i32
  store i32 %conv, ptr @i, align 4
  %1 = load i8, ptr @sc2, align 1
  %conv1 = sext i8 %1 to i32
  store i32 %conv1, ptr @j, align 4
; mips32r2:  lbu  $[[REG1:[0-9]+]], 0(${{[0-9]+}})
; mips32r2:  seb  ${{[0-9]+}}, $[[REG1]]
; mips32:  lbu  $[[REG1:[0-9]+]], 0(${{[0-9]+}})
; mips32:    sll  $[[REG2:[0-9]+]], $[[REG1]], 24
; mips32:    sra  ${{[0-9]+}}, $[[REG2]], 24

  ret void
; CHECK:  .end  _Z4sc_iv
}

; Function Attrs: nounwind
define void @_Z4us_iv()  {
entry:
; CHECK-LABEL:  .ent  _Z4us_iv
  %0 = load i16, ptr @us1, align 2
  %conv = zext i16 %0 to i32
  store i32 %conv, ptr @i, align 4
  %1 = load i16, ptr @us2, align 2
  %conv1 = zext i16 %1 to i32
  store i32 %conv1, ptr @j, align 4
  ret void
; CHECK:  lhu  $[[REG1:[0-9]+]], 0(${{[0-9]+}})
; CHECK:  andi  ${{[0-9]+}}, $[[REG1]], 65535
; CHECK:  .end  _Z4us_iv
}

; Function Attrs: nounwind
define void @_Z4ss_iv()  {
entry:
; mips32r2-LABEL:  .ent  _Z4ss_iv
; mips32=LABEL:  .ent  _Z4ss_iv

  %0 = load i16, ptr @ss1, align 2
  %conv = sext i16 %0 to i32
  store i32 %conv, ptr @i, align 4
  %1 = load i16, ptr @ss2, align 2
  %conv1 = sext i16 %1 to i32
  store i32 %conv1, ptr @j, align 4
; mips32r2:  lhu  $[[REG1:[0-9]+]], 0(${{[0-9]+}})
; mips32r2:  seh  ${{[0-9]+}}, $[[REG1]]
; mips32:    lhu  $[[REG1:[0-9]+]], 0(${{[0-9]+}})
; mips32:    sll  $[[REG2:[0-9]+]], $[[REG1]], 16
; mips32:    sra  ${{[0-9]+}}, $[[REG2]], 16

  ret void
; CHECK:  .end  _Z4ss_iv
}

; Function Attrs: nounwind
define void @_Z4b_ssv()  {
entry:
; CHECK-LABEL:  .ent  _Z4b_ssv
  %0 = load i8, ptr @b2, align 1
  %tobool = trunc i8 %0 to i1
  %conv = zext i1 %tobool to i16
  store i16 %conv, ptr @ssi, align 2
  ret void
; CHECK:  lbu  $[[REG1:[0-9]+]], 0(${{[0-9]+}})
; CHECK:  andi  ${{[0-9]+}}, $[[REG1]], 1
; CHECK:  .end  _Z4b_ssv
}

; Function Attrs: nounwind
define void @_Z5uc_ssv()  {
entry:
; CHECK-LABEL:  .ent  _Z5uc_ssv
  %0 = load i8, ptr @uc1, align 1
  %conv = zext i8 %0 to i16
  store i16 %conv, ptr @ssi, align 2
  %1 = load i8, ptr @uc2, align 1
  %conv1 = zext i8 %1 to i16
; CHECK:   lbu  $[[REG1:[0-9]+]], 0(${{[0-9]+}})
; CHECK:  andi  ${{[0-9]+}}, $[[REG1]], 255

  store i16 %conv1, ptr @ssj, align 2
  ret void
; CHECK:  .end  _Z5uc_ssv
}

; Function Attrs: nounwind
define void @_Z5sc_ssv()  {
entry:
; mips32r2-LABEL:  .ent  _Z5sc_ssv
; mips32-LABEL:  .ent  _Z5sc_ssv
  %0 = load i8, ptr @sc1, align 1
  %conv = sext i8 %0 to i16
  store i16 %conv, ptr @ssi, align 2
  %1 = load i8, ptr @sc2, align 1
  %conv1 = sext i8 %1 to i16
  store i16 %conv1, ptr @ssj, align 2
; mips32r2:  lbu  $[[REG1:[0-9]+]], 0(${{[0-9]+}})
; mips32r2:  seb  ${{[0-9]+}}, $[[REG1]]
; mips32:  lbu  $[[REG1:[0-9]+]], 0(${{[0-9]+}})
; mips32:    sll  $[[REG2:[0-9]+]], $[[REG1]], 24
; mips32:    sra  ${{[0-9]+}}, $[[REG2]], 24

  ret void
; CHECK:  .end  _Z5sc_ssv
}

