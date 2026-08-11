; Test that sub-64-bit integer formal arguments and small struct formal
; arguments are correctly sign/zero-extended by the callee in the z/OS
; XPLINK64 calling convention.
;
; The XPLINK64 ABI does not require the caller to have extended the upper bits
; of a GPR for named (non-variadic) integer arguments.  The callee therefore
; must perform its own extension when it needs the full 64-bit value.
;
; Integer formals (CCIfExtend -> CC_XPLINK_Promote_i32 with isFormalArgLowering=true):
;   For i8/i16 formals, CC_XPLINK_Promote_i32 keeps LocVT=i32 (GR32 live-in).
;   LowerFormalArguments truncates the GR32 to the true i8/i16 type, so the
;   subsequent sign/zero-extend to i64 selects the narrow register-extend
;   instructions (LGBR/LGHR/LLGCR/LLGHR) matching XL compiler output.
;   For i32 formals, LocVT=i64 (GR64), so LGFR is used as before.
;
;   jbyte    (i8  signext)  lgbr  -- sign-extend byte register to 64 bits
;   jboolean (i8  zeroext)  llgcr -- zero-extend char register to 64 bits
;   jshort   (i16 signext)  lghr  -- sign-extend halfword register to 64 bits
;   jchar    (i16 zeroext)  llghr -- zero-extend halfword register to 64 bits
;   jint     (i32 signext)  lgfr  -- sign-extend word register to 64 bits
;
; With optnone (alloca spill path), the reload uses the natural-width
; sign/zero-extend load instruction (lgb/stc+lgb).
;
; Small struct formals (coerced to i64, no CCIfExtend):
;   Struct member occupies the high bits of the GPR (big-endian).  The callee
;   must shift down to recover the signed value via srag (arithmetic shift).
;   S8  { signed char  }  srag 1,1,56
;   S16 { short        }  srag 1,1,48
;   S32 { int          }  srag 1,1,32
;
; Source:
;   void print(long long);
;   typedef signed char jbyte;  typedef unsigned char jboolean;
;   typedef short jshort;       typedef unsigned short jchar;
;   typedef int   jint;
;   struct S8 { signed char x; };  struct S16 { short x; };  struct S32 { int x; };
;   void callee_jbyte   (jbyte    c) { print((long long)c); }
;   void callee_jboolean(jboolean c) { print((long long)c); }
;   void callee_jshort  (jshort   c) { print((long long)c); }
;   void callee_jchar   (jchar    c) { print((long long)c); }
;   void callee_jint    (jint     c) { print((long long)c); }
;   void callee_jbyte_optnone(jbyte c) { print((long long)c); }  // optnone
;   void callee_s8 (struct S8  s)   { print((long long)s.x); }
;   void callee_s16(struct S16 s)   { print((long long)s.x); }
;   void callee_s32(struct S32 s)   { print((long long)s.x); }
;
; RUN: llc -mtriple=s390x-ibm-zos -O3 < %s | FileCheck %s --check-prefix=OPT
; RUN: llc -mtriple=s390x-ibm-zos -O0 < %s | FileCheck %s --check-prefix=NOOPT

;----------------------------------------------------------------------------
; jbyte (i8 signext): callee sign-extends incoming R1 (GR32) to 64 bits.
; At -O3: lgbr 1,1 (register form) or lgb (memory form, optimizer may spill).
; At -O0: lr copies R1 to R0, then lgbr 1,0.
;----------------------------------------------------------------------------
; OPT-LABEL: @callee_jbyte
; OPT:       L#end_of_prologue0
; OPT:        lgb
; OPT:        basr 7,6

; NOOPT-LABEL: @callee_jbyte
; NOOPT:       L#end_of_prologue0
; NOOPT:        lgbr 1,
; NOOPT:        basr 7,6

define hidden void @callee_jbyte(i8 noundef signext %c) local_unnamed_addr {
entry:
  %conv = sext i8 %c to i64
  tail call void @print(i64 noundef %conv)
  ret void
}

;----------------------------------------------------------------------------
; jboolean (i8 zeroext): callee zero-extends incoming R1 (GR32) to 64 bits.
; At -O3: llgcr 1,1 or llgc (memory form).
; At -O0: lr + llgcr 1,0.
;----------------------------------------------------------------------------
; OPT-LABEL: @callee_jboolean
; OPT:       L#end_of_prologue1
; OPT:        llgc
; OPT:        basr 7,6

; NOOPT-LABEL: @callee_jboolean
; NOOPT:       L#end_of_prologue1
; NOOPT:        llgcr 1,
; NOOPT:        basr 7,6

define hidden void @callee_jboolean(i8 noundef zeroext %c) local_unnamed_addr {
entry:
  %conv = zext i8 %c to i64
  tail call void @print(i64 noundef %conv)
  ret void
}

;----------------------------------------------------------------------------
; jshort (i16 signext): callee sign-extends incoming R1 (GR32) to 64 bits.
; At -O3: lghr 1,1 or lgh (memory form).
; At -O0: lr + lghr 1,0.
;----------------------------------------------------------------------------
; OPT-LABEL: @callee_jshort
; OPT:       L#end_of_prologue2
; OPT:        lgh
; OPT:        basr 7,6

; NOOPT-LABEL: @callee_jshort
; NOOPT:       L#end_of_prologue2
; NOOPT:        lghr 1,
; NOOPT:        basr 7,6

define hidden void @callee_jshort(i16 noundef signext %c) local_unnamed_addr {
entry:
  %conv = sext i16 %c to i64
  tail call void @print(i64 noundef %conv)
  ret void
}

;----------------------------------------------------------------------------
; jchar (i16 zeroext): callee zero-extends incoming R1 (GR32) to 64 bits.
; At -O3: llghr 1,1 or llgh (memory form).
; At -O0: lr + llghr 1,0.
;----------------------------------------------------------------------------
; OPT-LABEL: @callee_jchar
; OPT:       L#end_of_prologue3
; OPT:        llgh
; OPT:        basr 7,6

; NOOPT-LABEL: @callee_jchar
; NOOPT:       L#end_of_prologue3
; NOOPT:        llghr 1,
; NOOPT:        basr 7,6

define hidden void @callee_jchar(i16 noundef zeroext %c) local_unnamed_addr {
entry:
  %conv = zext i16 %c to i64
  tail call void @print(i64 noundef %conv)
  ret void
}

;----------------------------------------------------------------------------
; jint (i32 signext): callee sign-extends incoming R1 (GR64) to 64 bits.
; At -O3: lgfr 1,1.
; At -O0: lr + lgfr 1,0.
;----------------------------------------------------------------------------
; OPT-LABEL: @callee_jint
; OPT:       L#end_of_prologue4
; OPT:        lgfr 1,1
; OPT:        basr 7,6

; NOOPT-LABEL: @callee_jint
; NOOPT:       L#end_of_prologue4
; NOOPT:        lgfr 1,
; NOOPT:        basr 7,6

define hidden void @callee_jint(i32 noundef signext %c) local_unnamed_addr {
entry:
  %conv = sext i32 %c to i64
  tail call void @print(i64 noundef %conv)
  ret void
}

;----------------------------------------------------------------------------
; jbyte optnone: value spilled to alloca.  The reload uses lgb (sign-extend
; byte load) rather than lgbr, because the value passes through memory.
; At both -O3 and -O0: stc+lgb.  The stc source register may be 0 or 1.
;----------------------------------------------------------------------------
; OPT-LABEL: @callee_jbyte_optnone
; OPT:       L#end_of_prologue5
; OPT:        stc {{[01]}},{{[0-9]+}}(4)
; OPT:        lgb 1,{{[0-9]+}}(4)
; OPT:        basr 7,6

; NOOPT-LABEL: @callee_jbyte_optnone
; NOOPT:       L#end_of_prologue5
; NOOPT:        stc {{[01]}},{{[0-9]+}}(4)
; NOOPT:        lgb 1,{{[0-9]+}}(4)
; NOOPT:        basr 7,6

define hidden void @callee_jbyte_optnone(i8 noundef signext %c) noinline optnone {
entry:
  %c.addr = alloca i8, align 1
  store i8 %c, ptr %c.addr, align 1
  %0 = load i8, ptr %c.addr, align 1
  %conv = sext i8 %0 to i64
  call void @print(i64 noundef %conv)
  ret void
}

;----------------------------------------------------------------------------
; struct S8 { signed char x } coerced to i64 (struct, no CCIfExtend).
; The byte occupies the high bits; srag 1,1,56 extracts and sign-extends.
;----------------------------------------------------------------------------
; OPT-LABEL: @callee_s8
; OPT:       L#end_of_prologue6
; OPT:        srag 1,1,56
; OPT:        basr 7,6

; NOOPT-LABEL: @callee_s8
; NOOPT:       L#end_of_prologue6
; NOOPT:        srag 1,1,56
; NOOPT:        basr 7,6

define hidden void @callee_s8(i64 %s.coerce) local_unnamed_addr {
entry:
  %conv = ashr i64 %s.coerce, 56
  tail call void @print(i64 noundef %conv)
  ret void
}

;----------------------------------------------------------------------------
; struct S16 { short x } coerced to i64.
; The halfword occupies the high bits; srag 1,1,48 extracts and sign-extends.
;----------------------------------------------------------------------------
; OPT-LABEL: @callee_s16
; OPT:       L#end_of_prologue7
; OPT:        srag 1,1,48
; OPT:        basr 7,6

; NOOPT-LABEL: @callee_s16
; NOOPT:       L#end_of_prologue7
; NOOPT:        srag 1,1,48
; NOOPT:        basr 7,6

define hidden void @callee_s16(i64 %s.coerce) local_unnamed_addr {
entry:
  %conv = ashr i64 %s.coerce, 48
  tail call void @print(i64 noundef %conv)
  ret void
}

;----------------------------------------------------------------------------
; struct S32 { int x } coerced to i64.
; The word occupies the high bits; srag 1,1,32 extracts and sign-extends.
;----------------------------------------------------------------------------
; OPT-LABEL: @callee_s32
; OPT:       L#end_of_prologue8
; OPT:        srag 1,1,32
; OPT:        basr 7,6

; NOOPT-LABEL: @callee_s32
; NOOPT:       L#end_of_prologue8
; NOOPT:        srag 1,1,32
; NOOPT:        basr 7,6

define hidden void @callee_s32(i64 %s.coerce) local_unnamed_addr {
entry:
  %conv = ashr i64 %s.coerce, 32
  tail call void @print(i64 noundef %conv)
  ret void
}

declare void @print(i64 noundef)
