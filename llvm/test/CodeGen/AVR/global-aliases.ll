; RUN: llc < %s -mtriple=avr -mcpu=atxmega384c3 | FileCheck %s --check-prefixes=MEGA
; RUN: llc < %s -mtriple=avr -mcpu=attiny40 | FileCheck %s --check-prefixes=TINY

; MEGA: __tmp_reg__ = 0
; MEGA: __zero_reg__ = 1
; MEGA: __SREG__ = 63
; MEGA: __SP_H__ = 62
; MEGA: __SP_L__ = 61
; MEGA: __EIND__ = 60
; MEGA: __RAMPZ__ = 59

; TINY:     __tmp_reg__ = 16
; TINY:     __zero_reg__ = 17
; TINY:     __SREG__ = 63
; TINY-NOT: __SP_H__ = 62
; TINY:     __SP_L__ = 61
; TINY-NOT: __EIND__ = 60
; TINY-NOT: __RAMPZ__ = 59
