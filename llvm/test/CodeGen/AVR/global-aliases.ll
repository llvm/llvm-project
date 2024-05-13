; RUN: llc < %s -mtriple=avr -mcpu=atxmega384c3 | FileCheck %s --check-prefixes=MEGA
; RUN: llc < %s -mtriple=avr -mcpu=attiny40 | FileCheck %s --check-prefixes=TINY

; MEGA: .set __tmp_reg__, 0
; MEGA: .set __zero_reg__, 1
; MEGA: .set __SREG__, 63
; MEGA: .set __SP_H__, 62
; MEGA: .set __SP_L__, 61
; MEGA: .set __EIND__, 60
; MEGA: .set __RAMPZ__, 59

; TINY:     .set __tmp_reg__, 16
; TINY:     .set __zero_reg__, 17
; TINY:     .set __SREG__, 63
; TINY-NOT: .set __SP_H__, 62
; TINY:     .set __SP_L__, 61
; TINY-NOT: .set __EIND__, 60
; TINY-NOT: .set __RAMPZ__, 59
