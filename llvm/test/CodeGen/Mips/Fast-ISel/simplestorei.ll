; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32r2 \
; RUN:     < %s | FileCheck %s
; RUN: llc -march=mipsel -relocation-model=pic -O0 -fast-isel-abort=3 -mcpu=mips32 \
; RUN:     < %s | FileCheck %s

@ijk = external global i32

; Function Attrs: nounwind
define void @si2_1() #0 {
entry:
  store i32 32767, i32* @ijk, align 4
; CHECK:        .ent    si2_1
; CHECK:        addiu   $[[REG1:[0-9]+]], $zero, 32767
; CHECK:        lw      $[[REG2:[0-9]+]], %got(ijk)(${{[0-9]+}})
; CHECK:        sw      $[[REG1]], 0($[[REG2]])

  ret void
}

; Function Attrs: nounwind
define void @si2_2() #0 {
entry:
  store i32 -32768, i32* @ijk, align 4
; CHECK:        .ent    si2_2
; CHECK:        lui     $[[REG1:[0-9]+]], 65535
; CHECK:        ori     $[[REG2:[0-9]+]], $[[REG1]], 32768
; CHECK:        lw      $[[REG3:[0-9]+]], %got(ijk)(${{[0-9]+}})
; CHECK:        sw      $[[REG2]], 0($[[REG3]])
  ret void
}

; Function Attrs: nounwind
define void @ui2_1() #0 {
entry:
  store i32 65535, i32* @ijk, align 4
; CHECK:        .ent    ui2_1
; CHECK:        ori     $[[REG1:[0-9]+]], $zero, 65535
; CHECK:        lw      $[[REG2:[0-9]+]], %got(ijk)(${{[0-9]+}})
; CHECK:        sw      $[[REG1]], 0($[[REG2]])
  ret void
}

; Function Attrs: nounwind
define void @ui4_1() #0 {
entry:
  store i32 983040, i32* @ijk, align 4
; CHECK:        .ent    ui4_1
; CHECK:        lui     $[[REG1:[0-9]+]], 15
; CHECK:        lw      $[[REG2:[0-9]+]], %got(ijk)(${{[0-9]+}})
; CHECK:        sw      $[[REG1]], 0($[[REG2]])
  ret void
}

; Function Attrs: nounwind
define void @ui4_2() #0 {
entry:
  store i32 719566, i32* @ijk, align 4
; CHECK:        .ent    ui4_2
; CHECK:        lui	$[[REG1:[0-9]+]], 10
; CHECK: 	ori	$[[REG1]], $[[REG1]], 64206
; CHECK: 	lw	$[[REG2:[0-9]+]], %got(ijk)(${{[0-9]+}})
; CHECK: 	sw	$[[REG1]], 0($[[REG2]])
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }


