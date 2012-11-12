; This test makes sure that these instructions are properly eliminated.
;
; RUN: opt < %s -instcombine -S | \
; RUN:    not grep "tobool"
; END.
define i32 @main(i32 %argc, i8** %argv) nounwind ssp {
entry:
  %and = and i32 %argc, 1                         ; <i32> [#uses=1]
  %tobool = icmp ne i32 %and, 0                   ; <i1> [#uses=1]
  %and2 = and i32 %argc, 2                        ; <i32> [#uses=1]
  %tobool3 = icmp ne i32 %and2, 0                 ; <i1> [#uses=1]
  %or.cond = and i1 %tobool, %tobool3             ; <i1> [#uses=1]
  %retval.0 = select i1 %or.cond, i32 2, i32 1    ; <i32> [#uses=1]
  ret i32 %retval.0
}

define i32 @main2(i32 %argc, i8** nocapture %argv) nounwind readnone ssp {
entry:
  %and = and i32 %argc, 1                         ; <i32> [#uses=1]
  %tobool = icmp eq i32 %and, 0                   ; <i1> [#uses=1]
  %and2 = and i32 %argc, 2                        ; <i32> [#uses=1]
  %tobool3 = icmp eq i32 %and2, 0                 ; <i1> [#uses=1]
  %or.cond = or i1 %tobool, %tobool3              ; <i1> [#uses=1]
  %storemerge = select i1 %or.cond, i32 0, i32 1  ; <i32> [#uses=1]
  ret i32 %storemerge
}

; tests to check combining (icmp eq (A & B), C) & (icmp eq (A & D), E)
; tests to check if (icmp eq (A & B), 0) is treated like (icmp eq (A & B), B)
; if B is a single bit constant

; (icmp eq (A & B), 0) & (icmp eq (A & D), 0) -> (icmp eq (A & (B|D)), 0)
define i32 @main3(i32 %argc, i8** nocapture %argv) nounwind readnone ssp {
entry:
  %and = and i32 %argc, 7                         ; <i32> [#uses=1]
  %tobool = icmp eq i32 %and, 0                   ; <i1> [#uses=1]
  %and2 = and i32 %argc, 48                       ; <i32> [#uses=1]
  %tobool3 = icmp eq i32 %and2, 0                 ; <i1> [#uses=1]
  %and.cond = and i1 %tobool, %tobool3            ; <i1> [#uses=1]
  %storemerge = select i1 %and.cond, i32 0, i32 1 ; <i32> [#uses=1]
  ret i32 %storemerge
}

define i32 @main3b(i32 %argc, i8** nocapture %argv) nounwind readnone ssp {
entry:
  %and = and i32 %argc, 7                         ; <i32> [#uses=1]
  %tobool = icmp eq i32 %and, 0                   ; <i1> [#uses=1]
  %and2 = and i32 %argc, 16                       ; <i32> [#uses=1]
  %tobool3 = icmp ne i32 %and2, 16                 ; <i1> [#uses=1]
  %and.cond = and i1 %tobool, %tobool3            ; <i1> [#uses=1]
  %storemerge = select i1 %and.cond, i32 0, i32 1 ; <i32> [#uses=1]
  ret i32 %storemerge
}

define i32 @main3e_like(i32 %argc, i32 %argc2, i32 %argc3, i8** nocapture %argv)
           nounwind readnone ssp {
entry:
  %and = and i32 %argc, %argc2                    ; <i32> [#uses=1]
  %tobool = icmp eq i32 %and, 0                   ; <i1> [#uses=1]
  %and2 = and i32 %argc, %argc3                   ; <i32> [#uses=1]
  %tobool3 = icmp eq i32 %and2, 0                 ; <i1> [#uses=1]
  %and.cond = and i1 %tobool, %tobool3            ; <i1> [#uses=1]
  %storemerge = select i1 %and.cond, i32 0, i32 1 ; <i32> [#uses=1]
  ret i32 %storemerge
}

; (icmp ne (A & B), 0) | (icmp ne (A & D), 0) -> (icmp ne (A & (B|D)), 0)
define i32 @main3c(i32 %argc, i8** nocapture %argv) nounwind readnone ssp {
entry:
  %and = and i32 %argc, 7                         ; <i32> [#uses=1]
  %tobool = icmp ne i32 %and, 0                   ; <i1> [#uses=1]
  %and2 = and i32 %argc, 48                       ; <i32> [#uses=1]
  %tobool3 = icmp ne i32 %and2, 0                 ; <i1> [#uses=1]
  %or.cond = or i1 %tobool, %tobool3              ; <i1> [#uses=1]
  %storemerge = select i1 %or.cond, i32 0, i32 1  ; <i32> [#uses=1]
  ret i32 %storemerge
}

define i32 @main3d(i32 %argc, i8** nocapture %argv) nounwind readnone ssp {
entry:
  %and = and i32 %argc, 7                         ; <i32> [#uses=1]
  %tobool = icmp ne i32 %and, 0                   ; <i1> [#uses=1]
  %and2 = and i32 %argc, 16                       ; <i32> [#uses=1]
  %tobool3 = icmp eq i32 %and2, 16                ; <i1> [#uses=1]
  %or.cond = or i1 %tobool, %tobool3              ; <i1> [#uses=1]
  %storemerge = select i1 %or.cond, i32 0, i32 1  ; <i32> [#uses=1]
  ret i32 %storemerge
}

define i32 @main3f_like(i32 %argc, i32 %argc2, i32 %argc3, i8** nocapture %argv)
           nounwind readnone ssp {
entry:
  %and = and i32 %argc, %argc2                    ; <i32> [#uses=1]
  %tobool = icmp ne i32 %and, 0                   ; <i1> [#uses=1]
  %and2 = and i32 %argc, %argc3                   ; <i32> [#uses=1]
  %tobool3 = icmp ne i32 %and2, 0                 ; <i1> [#uses=1]
  %or.cond = or i1 %tobool, %tobool3              ; <i1> [#uses=1]
  %storemerge = select i1 %or.cond, i32 0, i32 1  ; <i32> [#uses=1]
  ret i32 %storemerge
}

; (icmp eq (A & B), B) & (icmp eq (A & D), D) -> (icmp eq (A & (B|D)), (B|D))
define i32 @main4(i32 %argc, i8** nocapture %argv) nounwind readnone ssp {
entry:
  %and = and i32 %argc, 7                         ; <i32> [#uses=1]
  %tobool = icmp eq i32 %and, 7                   ; <i1> [#uses=1]
  %and2 = and i32 %argc, 48                       ; <i32> [#uses=1]
  %tobool3 = icmp eq i32 %and2, 48                ; <i1> [#uses=1]
  %and.cond = and i1 %tobool, %tobool3            ; <i1> [#uses=1]
  %storemerge = select i1 %and.cond, i32 0, i32 1 ; <i32> [#uses=1]
  ret i32 %storemerge
}

define i32 @main4b(i32 %argc, i8** nocapture %argv) nounwind readnone ssp {
entry:
  %and = and i32 %argc, 7                         ; <i32> [#uses=1]
  %tobool = icmp eq i32 %and, 7                   ; <i1> [#uses=1]
  %and2 = and i32 %argc, 16                       ; <i32> [#uses=1]
  %tobool3 = icmp ne i32 %and2, 0                 ; <i1> [#uses=1]
  %and.cond = and i1 %tobool, %tobool3            ; <i1> [#uses=1]
  %storemerge = select i1 %and.cond, i32 0, i32 1 ; <i32> [#uses=1]
  ret i32 %storemerge
}

define i32 @main4e_like(i32 %argc, i32 %argc2, i32 %argc3, i8** nocapture %argv)
           nounwind readnone ssp {
entry:
  %and = and i32 %argc, %argc2                    ; <i32> [#uses=1]
  %tobool = icmp eq i32 %and, %argc2              ; <i1> [#uses=1]
  %and2 = and i32 %argc, %argc3                   ; <i32> [#uses=1]
  %tobool3 = icmp eq i32 %and2, %argc3            ; <i1> [#uses=1]
  %and.cond = and i1 %tobool, %tobool3            ; <i1> [#uses=1]
  %storemerge = select i1 %and.cond, i32 0, i32 1 ; <i32> [#uses=1]
  ret i32 %storemerge
}

; (icmp ne (A & B), B) | (icmp ne (A & D), D) -> (icmp ne (A & (B|D)), (B|D))
define i32 @main4c(i32 %argc, i8** nocapture %argv) nounwind readnone ssp {
entry:
  %and = and i32 %argc, 7                         ; <i32> [#uses=1]
  %tobool = icmp ne i32 %and, 7                   ; <i1> [#uses=1]
  %and2 = and i32 %argc, 48                       ; <i32> [#uses=1]
  %tobool3 = icmp ne i32 %and2, 48                ; <i1> [#uses=1]
  %or.cond = or i1 %tobool, %tobool3              ; <i1> [#uses=1]
  %storemerge = select i1 %or.cond, i32 0, i32 1  ; <i32> [#uses=1]
  ret i32 %storemerge
}

define i32 @main4d(i32 %argc, i8** nocapture %argv) nounwind readnone ssp {
entry:
  %and = and i32 %argc, 7                         ; <i32> [#uses=1]
  %tobool = icmp ne i32 %and, 7                   ; <i1> [#uses=1]
  %and2 = and i32 %argc, 16                       ; <i32> [#uses=1]
  %tobool3 = icmp eq i32 %and2, 0                 ; <i1> [#uses=1]
  %or.cond = or i1 %tobool, %tobool3              ; <i1> [#uses=1]
  %storemerge = select i1 %or.cond, i32 0, i32 1  ; <i32> [#uses=1]
  ret i32 %storemerge
}

define i32 @main4f_like(i32 %argc, i32 %argc2, i32 %argc3, i8** nocapture %argv)
           nounwind readnone ssp {
entry:
  %and = and i32 %argc, %argc2                    ; <i32> [#uses=1]
  %tobool = icmp ne i32 %and, %argc2              ; <i1> [#uses=1]
  %and2 = and i32 %argc, %argc3                   ; <i32> [#uses=1]
  %tobool3 = icmp ne i32 %and2, %argc3            ; <i1> [#uses=1]
  %or.cond = or i1 %tobool, %tobool3              ; <i1> [#uses=1]
  %storemerge = select i1 %or.cond, i32 0, i32 1  ; <i32> [#uses=1]
  ret i32 %storemerge
}

; (icmp eq (A & B), A) & (icmp eq (A & D), A) -> (icmp eq (A & (B&D)), A)
define i32 @main5_like(i32 %argc, i32 %argc2, i8** nocapture %argv)
           nounwind readnone ssp {
entry:
  %and = and i32 %argc, 7                         ; <i32> [#uses=1]
  %tobool = icmp eq i32 %and, 7                   ; <i1> [#uses=1]
  %and2 = and i32 %argc2, 7                       ; <i32> [#uses=1]
  %tobool3 = icmp eq i32 %and2, 7                 ; <i1> [#uses=1]
  %and.cond = and i1 %tobool, %tobool3            ; <i1> [#uses=1]
  %storemerge = select i1 %and.cond, i32 0, i32 1 ; <i32> [#uses=1]
  ret i32 %storemerge
}

define i32 @main5e_like(i32 %argc, i32 %argc2, i32 %argc3, i8** nocapture %argv)
           nounwind readnone ssp {
entry:
  %and = and i32 %argc, %argc2                    ; <i32> [#uses=1]
  %tobool = icmp eq i32 %and, %argc               ; <i1> [#uses=1]
  %and2 = and i32 %argc, %argc3                   ; <i32> [#uses=1]
  %tobool3 = icmp eq i32 %and2, %argc             ; <i1> [#uses=1]
  %and.cond = and i1 %tobool, %tobool3            ; <i1> [#uses=1]
  %storemerge = select i1 %and.cond, i32 0, i32 1 ; <i32> [#uses=1]
  ret i32 %storemerge
}

; (icmp ne (A & B), A) | (icmp ne (A & D), A) -> (icmp ne (A & (B&D)), A)
define i32 @main5c_like(i32 %argc, i32 %argc2, i8** nocapture %argv)
           nounwind readnone ssp {
entry:
  %and = and i32 %argc, 7                         ; <i32> [#uses=1]
  %tobool = icmp ne i32 %and, 7                   ; <i1> [#uses=1]
  %and2 = and i32 %argc2, 7                       ; <i32> [#uses=1]
  %tobool3 = icmp ne i32 %and2, 7                 ; <i1> [#uses=1]
  %or.cond = or i1 %tobool, %tobool3              ; <i1> [#uses=1]
  %storemerge = select i1 %or.cond, i32 0, i32 1  ; <i32> [#uses=1]
  ret i32 %storemerge
}

define i32 @main5f_like(i32 %argc, i32 %argc2, i32 %argc3, i8** nocapture %argv)
           nounwind readnone ssp {
entry:
  %and = and i32 %argc, %argc2                    ; <i32> [#uses=1]
  %tobool = icmp ne i32 %and, %argc               ; <i1> [#uses=1]
  %and2 = and i32 %argc, %argc3                   ; <i32> [#uses=1]
  %tobool3 = icmp ne i32 %and2, %argc             ; <i1> [#uses=1]
  %or.cond = or i1 %tobool, %tobool3              ; <i1> [#uses=1]
  %storemerge = select i1 %or.cond, i32 0, i32 1  ; <i32> [#uses=1]
  ret i32 %storemerge
}

; (icmp eq (A & B), C) & (icmp eq (A & D), E) -> (icmp eq (A & (B|D)), (C|E))
; if B, C, D, E are constant, and it's possible
define i32 @main6(i32 %argc, i8** nocapture %argv) nounwind readnone ssp {
entry:
  %and = and i32 %argc, 7                         ; <i32> [#uses=1]
  %tobool = icmp eq i32 %and, 3                   ; <i1> [#uses=1]
  %and2 = and i32 %argc, 48                       ; <i32> [#uses=1]
  %tobool3 = icmp eq i32 %and2, 16                ; <i1> [#uses=1]
  %and.cond = and i1 %tobool, %tobool3            ; <i1> [#uses=1]
  %storemerge = select i1 %and.cond, i32 0, i32 1 ; <i32> [#uses=1]
  ret i32 %storemerge
}

define i32 @main6b(i32 %argc, i8** nocapture %argv) nounwind readnone ssp {
entry:
  %and = and i32 %argc, 7                         ; <i32> [#uses=1]
  %tobool = icmp eq i32 %and, 3                   ; <i1> [#uses=1]
  %and2 = and i32 %argc, 16                       ; <i32> [#uses=1]
  %tobool3 = icmp ne i32 %and2, 0                 ; <i1> [#uses=1]
  %and.cond = and i1 %tobool, %tobool3            ; <i1> [#uses=1]
  %storemerge = select i1 %and.cond, i32 0, i32 1 ; <i32> [#uses=1]
  ret i32 %storemerge
}

; (icmp ne (A & B), C) | (icmp ne (A & D), E) -> (icmp ne (A & (B|D)), (C|E))
; if B, C, D, E are constant, and it's possible
define i32 @main6c(i32 %argc, i8** nocapture %argv) nounwind readnone ssp {
entry:
  %and = and i32 %argc, 7                         ; <i32> [#uses=1]
  %tobool = icmp ne i32 %and, 3                   ; <i1> [#uses=1]
  %and2 = and i32 %argc, 48                       ; <i32> [#uses=1]
  %tobool3 = icmp ne i32 %and2, 16                ; <i1> [#uses=1]
  %or.cond = or i1 %tobool, %tobool3              ; <i1> [#uses=1]
  %storemerge = select i1 %or.cond, i32 0, i32 1  ; <i32> [#uses=1]
  ret i32 %storemerge
}

define i32 @main6d(i32 %argc, i8** nocapture %argv) nounwind readnone ssp {
entry:
  %and = and i32 %argc, 7                         ; <i32> [#uses=1]
  %tobool = icmp ne i32 %and, 3                   ; <i1> [#uses=1]
  %and2 = and i32 %argc, 16                       ; <i32> [#uses=1]
  %tobool3 = icmp eq i32 %and2, 0                 ; <i1> [#uses=1]
  %or.cond = or i1 %tobool, %tobool3              ; <i1> [#uses=1]
  %storemerge = select i1 %or.cond, i32 0, i32 1  ; <i32> [#uses=1]
  ret i32 %storemerge
}

; test parameter permutations
; (B & A) == B & (D & A) == D
define i32 @main7a(i32 %argc, i32 %argc2, i32 %argc3, i8** nocapture %argv)
           nounwind readnone ssp {
entry:
  %and1 = and i32 %argc2, %argc                   ; <i32> [#uses=1]
  %tobool = icmp eq i32 %and1, %argc2              ; <i1> [#uses=1]
  %and2 = and i32 %argc3, %argc                   ; <i32> [#uses=1]
  %tobool3 = icmp eq i32 %and2, %argc3            ; <i1> [#uses=1]
  %and.cond = and i1 %tobool, %tobool3            ; <i1> [#uses=1]
  %storemerge = select i1 %and.cond, i32 0, i32 1 ; <i32> [#uses=1]
  ret i32 %storemerge
}

; B == (A & B) & D == (A & D)
define i32 @main7b(i32 %argc, i32 %argc2, i32 %argc3, i8** nocapture %argv)
           nounwind readnone ssp {
entry:
  %and1 = and i32 %argc, %argc2                   ; <i32> [#uses=1]
  %tobool = icmp eq i32 %argc2, %and1             ; <i1> [#uses=1]
  %and2 = and i32 %argc, %argc3                   ; <i32> [#uses=1]
  %tobool3 = icmp eq i32 %argc3, %and2            ; <i1> [#uses=1]
  %and.cond = and i1 %tobool, %tobool3            ; <i1> [#uses=1]
  %storemerge = select i1 %and.cond, i32 0, i32 1 ; <i32> [#uses=1]
  ret i32 %storemerge
}

; B == (B & A) & D == (D & A)
define i32 @main7c(i32 %argc, i32 %argc2, i32 %argc3, i8** nocapture %argv)
           nounwind readnone ssp {
entry:
  %and1 = and i32 %argc2, %argc                   ; <i32> [#uses=1]
  %tobool = icmp eq i32 %argc2, %and1             ; <i1> [#uses=1]
  %and2 = and i32 %argc3, %argc                   ; <i32> [#uses=1]
  %tobool3 = icmp eq i32 %argc3, %and2            ; <i1> [#uses=1]
  %and.cond = and i1 %tobool, %tobool3            ; <i1> [#uses=1]
  %storemerge = select i1 %and.cond, i32 0, i32 1 ; <i32> [#uses=1]
  ret i32 %storemerge
}

; (A & (B & C)) == (B & C) & (A & (D & E)) == (D & E)
define i32 @main7d(i32 %argc, i32 %argc2, i32 %argc3,
                   i32 %argc4, i32 %argc5, i8** nocapture %argv)
           nounwind readnone ssp {
entry:
  %bc = and i32 %argc2, %argc4                    ; <i32> [#uses=1]
  %de = and i32 %argc3, %argc5                    ; <i32> [#uses=1]
  %and1 = and i32 %argc, %bc                      ; <i32> [#uses=1]
  %tobool = icmp eq i32 %and1, %bc                ; <i1> [#uses=1]
  %and2 = and i32 %argc, %de                      ; <i32> [#uses=1]
  %tobool3 = icmp eq i32 %and2, %de               ; <i1> [#uses=1]
  %and.cond = and i1 %tobool, %tobool3            ; <i1> [#uses=1]
  %storemerge = select i1 %and.cond, i32 0, i32 1 ; <i32> [#uses=1]
  ret i32 %storemerge
}

; ((B & C) & A) == (B & C) & ((D & E) & A) == (D & E)
define i32 @main7e(i32 %argc, i32 %argc2, i32 %argc3,
                   i32 %argc4, i32 %argc5, i8** nocapture %argv)
           nounwind readnone ssp {
entry:
  %bc = and i32 %argc2, %argc4                    ; <i32> [#uses=1]
  %de = and i32 %argc3, %argc5                    ; <i32> [#uses=1]
  %and1 = and i32 %bc, %argc                      ; <i32> [#uses=1]
  %tobool = icmp eq i32 %and1, %bc                ; <i1> [#uses=1]
  %and2 = and i32 %de, %argc                      ; <i32> [#uses=1]
  %tobool3 = icmp eq i32 %and2, %de               ; <i1> [#uses=1]
  %and.cond = and i1 %tobool, %tobool3            ; <i1> [#uses=1]
  %storemerge = select i1 %and.cond, i32 0, i32 1 ; <i32> [#uses=1]
  ret i32 %storemerge
}

; (B & C) == (A & (B & C)) & (D & E) == (A & (D & E))
define i32 @main7f(i32 %argc, i32 %argc2, i32 %argc3,
                   i32 %argc4, i32 %argc5, i8** nocapture %argv)
           nounwind readnone ssp {
entry:
  %bc = and i32 %argc2, %argc4                    ; <i32> [#uses=1]
  %de = and i32 %argc3, %argc5                    ; <i32> [#uses=1]
  %and1 = and i32 %argc, %bc                      ; <i32> [#uses=1]
  %tobool = icmp eq i32 %bc, %and1                ; <i1> [#uses=1]
  %and2 = and i32 %argc, %de                      ; <i32> [#uses=1]
  %tobool3 = icmp eq i32 %de, %and2               ; <i1> [#uses=1]
  %and.cond = and i1 %tobool, %tobool3            ; <i1> [#uses=1]
  %storemerge = select i1 %and.cond, i32 0, i32 1 ; <i32> [#uses=1]
  ret i32 %storemerge
}

; (B & C) == ((B & C) & A) & (D & E) == ((D & E) & A)
define i32 @main7g(i32 %argc, i32 %argc2, i32 %argc3,
                   i32 %argc4, i32 %argc5, i8** nocapture %argv)
           nounwind readnone ssp {
entry:
  %bc = and i32 %argc2, %argc4                    ; <i32> [#uses=1]
  %de = and i32 %argc3, %argc5                    ; <i32> [#uses=1]
  %and1 = and i32 %bc, %argc                      ; <i32> [#uses=1]
  %tobool = icmp eq i32 %bc, %and1                ; <i1> [#uses=1]
  %and2 = and i32 %de, %argc                      ; <i32> [#uses=1]
  %tobool3 = icmp eq i32 %de, %and2               ; <i1> [#uses=1]
  %and.cond = and i1 %tobool, %tobool3            ; <i1> [#uses=1]
  %storemerge = select i1 %and.cond, i32 0, i32 1 ; <i32> [#uses=1]
  ret i32 %storemerge
}
