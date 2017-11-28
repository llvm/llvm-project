; RUN: llc -mtriple=x86_64-- -mcpu=core2 -fast-isel -enable-misched -misched=shuffle -misched-bottomup -verify-machineinstrs < %s
; RUN: llc -mtriple=x86_64-- -mcpu=core2 -fast-isel -enable-misched -misched=shuffle -misched-topdown -verify-machineinstrs < %s
; REQUIRES: asserts
;
; Test the LiveIntervals::handleMove() function.
;
; Moving the DIV32r instruction exercises the regunit update code because
; %edx has a live range into the function and is used by the DIV32r.
;
; Here sinking a kill + dead def:
; 144B -> 180B: DIV32r %vreg4, %eax<imp-def>, %edx<imp-def,dead>, %EFLAGS<imp-def,dead>, %eax<imp-use,kill>, %edx<imp-use>
;       %vreg4: [48r,144r:0)  0@48r
;         -->   [48r,180r:0)  0@48r
;       DH:     [0B,16r:0)[128r,144r:2)[144r,144d:1)  0@0B-phi 1@144r 2@128r
;         -->   [0B,16r:0)[128r,180r:2)[180r,180d:1)  0@0B-phi 1@180r 2@128r
;       DL:     [0B,16r:0)[128r,144r:2)[144r,144d:1)  0@0B-phi 1@144r 2@128r
;         -->   [0B,16r:0)[128r,180r:2)[180r,180d:1)  0@0B-phi 1@180r 2@128r
;
define i32 @f1(i32 %a, i32 %b, i32 %c) nounwind uwtable readnone ssp {
entry:
  %y = add i32 %c, 1
  %x = udiv i32 %b, %a
  %add = add nsw i32 %y, %x
  ret i32 %add
}

; Same as above, but moving a kill + live def:
; 144B -> 180B: DIV32r %vreg4, %eax<imp-def,dead>, %edx<imp-def>, %EFLAGS<imp-def,dead>, %eax<imp-use,kill>, %edx<imp-use>
;       %vreg4: [48r,144r:0)  0@48r
;         -->   [48r,180r:0)  0@48r
;       DH:     [0B,16r:0)[128r,144r:2)[144r,184r:1)  0@0B-phi 1@144r 2@128r
;         -->   [0B,16r:0)[128r,180r:2)[180r,184r:1)  0@0B-phi 1@180r 2@128r
;       DL:     [0B,16r:0)[128r,144r:2)[144r,184r:1)  0@0B-phi 1@144r 2@128r
;         -->   [0B,16r:0)[128r,180r:2)[180r,184r:1)  0@0B-phi 1@180r 2@128r
;
define i32 @f2(i32 %a, i32 %b, i32 %c, i32 %d) nounwind uwtable readnone ssp {
entry:
  %y = sub i32 %c, %d
  %x = urem i32 %b, %a
  %add = add nsw i32 %x, %y
  ret i32 %add
}

; Moving a use below the existing kill (%vreg5):
; Moving a tied virtual register def (%vreg11):
;
; 96B -> 120B: %vreg11<def,tied1> = SUB32rr %vreg11<tied0>, %vreg5
;       %vreg11:        [80r,96r:1)[96r,144r:0)  0@96r 1@80r
;            -->        [80r,120r:1)[120r,144r:0)  0@120r 1@80r
;       %vreg5:         [16r,112r:0)  0@16r
;            -->        [16r,120r:0)  0@16r
;
define i32 @f3(i32 %a, i32 %b) nounwind uwtable readnone ssp {
entry:
  %y = sub i32 %a, %b
  %x = add i32 %a, %b
  %r = mul i32 %x, %y
  ret i32 %r
}

; Move EFLAGS dead def across another def:
; handleMove 208B -> 36B: %edx<def> = MOV32r0 %EFLAGS<imp-def,dead>
;    EFLAGS:    [20r,20d:4)[160r,160d:3)[208r,208d:0)[224r,224d:1)[272r,272d:2)[304r,304d:5)  0@208r 1@224r 2@272r 3@160r 4@20r 5@304r
;         -->   [20r,20d:4)[36r,36d:0)[160r,160d:3)[224r,224d:1)[272r,272d:2)[304r,304d:5)  0@36r 1@224r 2@272r 3@160r 4@20r 5@304r
;
define i32 @f4(i32 %a, i32 %b, i32 %c, i32 %d) nounwind uwtable readnone ssp {
entry:
  %x = sub i32 %a, %b
  %y = sub i32 %b, %c
  %z = sub i32 %c, %d
  %r1 = udiv i32 %x, %y
  %r2 = mul i32 %z, %r1
  ret i32 %r2
}
