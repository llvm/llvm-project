; RUN: opt %loadPolly -polly-dependences -analyze < %s | FileCheck %s

;    void manyreductions(long *A) {
;      for (long i = 0; i < 1024; i++)
;        for (long j = 0; j < 1024; j++)
;          *A += 42;
;
;      for (long i = 0; i < 1024; i++)
;        for (long j = 0; j < 1024; j++)
;          *A += 42;
;
;      for (long i = 0; i < 1024; i++)
;        for (long j = 0; j < 1024; j++)
;          *A += 42;
;
;      for (long i = 0; i < 1024; i++)
;        for (long j = 0; j < 1024; j++)
;          *A += 42;
;
;      for (long i = 0; i < 1024; i++)
;        for (long j = 0; j < 1024; j++)
;          *A += 42;
;
;      for (long i = 0; i < 1024; i++)
;        for (long j = 0; j < 1024; j++)
;          *A += 42;
;
;      for (long i = 0; i < 1024; i++)
;        for (long j = 0; j < 1024; j++)
;          *A += 42;
;
;      for (long i = 0; i < 1024; i++)
;        for (long j = 0; j < 1024; j++)
;          *A += 42;
;
;      for (long i = 0; i < 1024; i++)
;        for (long j = 0; j < 1024; j++)
;          *A += 42;
;
;      for (long i = 0; i < 1024; i++)
;        for (long j = 0; j < 1024; j++)
;          *A += 42;
;
;      for (long i = 0; i < 1024; i++)
;        for (long j = 0; j < 1024; j++)
;          *A += 42;
;
;      for (long i = 0; i < 1024; i++)
;        for (long j = 0; j < 1024; j++)
;          *A += 42;
;
;      for (long i = 0; i < 1024; i++)
;        for (long j = 0; j < 1024; j++)
;          *A += 42;
;
;      for (long i = 0; i < 1024; i++)
;        for (long j = 0; j < 1024; j++)
;          *A += 42;
;    }

; CHECK: 	RAW dependences:
; CHECK-NEXT: 		{ Stmt_bb150[1023, 1023] -> Stmt_bb162[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb150[i0, i1] -> Stmt_bb162[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb150[1023, 1023] -> Stmt_bb162[0, 0]; Stmt_bb174[1023, 1023] -> Stmt_bb186[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb174[i0, i1] -> Stmt_bb186[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb174[1023, 1023] -> Stmt_bb186[0, 0]; Stmt_bb102[1023, 1023] -> Stmt_bb114[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb102[i0, i1] -> Stmt_bb114[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb102[1023, 1023] -> Stmt_bb114[0, 0]; Stmt_bb42[1023, 1023] -> Stmt_bb54[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb42[i0, i1] -> Stmt_bb54[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb42[1023, 1023] -> Stmt_bb54[0, 0]; Stmt_bb54[1023, 1023] -> Stmt_bb66[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb54[i0, i1] -> Stmt_bb66[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb54[1023, 1023] -> Stmt_bb66[0, 0]; Stmt_bb31[1023, 1023] -> Stmt_bb42[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb31[i0, i1] -> Stmt_bb42[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb31[1023, 1023] -> Stmt_bb42[0, 0]; Stmt_bb162[1023, 1023] -> Stmt_bb174[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb162[i0, i1] -> Stmt_bb174[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb162[1023, 1023] -> Stmt_bb174[0, 0]; Stmt_bb126[1023, 1023] -> Stmt_bb138[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb126[i0, i1] -> Stmt_bb138[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb126[1023, 1023] -> Stmt_bb138[0, 0]; Stmt_bb90[1023, 1023] -> Stmt_bb102[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb90[i0, i1] -> Stmt_bb102[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb90[1023, 1023] -> Stmt_bb102[0, 0]; Stmt_bb138[1023, 1023] -> Stmt_bb150[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb138[i0, i1] -> Stmt_bb150[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb138[1023, 1023] -> Stmt_bb150[0, 0]; Stmt_bb66[1023, 1023] -> Stmt_bb78[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb66[i0, i1] -> Stmt_bb78[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb66[1023, 1023] -> Stmt_bb78[0, 0]; Stmt_bb78[1023, 1023] -> Stmt_bb90[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb78[i0, i1] -> Stmt_bb90[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb78[1023, 1023] -> Stmt_bb90[0, 0]; Stmt_bb114[1023, 1023] -> Stmt_bb126[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb114[i0, i1] -> Stmt_bb126[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb114[1023, 1023] -> Stmt_bb126[0, 0] }
; CHECK-NEXT: 	WAR dependences:
; CHECK-NEXT:     { Stmt_bb150[1023, 1023] -> Stmt_bb162[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb150[i0, i1] -> Stmt_bb162[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb150[1023, 1023] -> Stmt_bb162[0, 0]; Stmt_bb174[1023, 1023] -> Stmt_bb186[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb174[i0, i1] -> Stmt_bb186[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb174[1023, 1023] -> Stmt_bb186[0, 0]; Stmt_bb102[1023, 1023] -> Stmt_bb114[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb102[i0, i1] -> Stmt_bb114[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb102[1023, 1023] -> Stmt_bb114[0, 0]; Stmt_bb42[1023, 1023] -> Stmt_bb54[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb42[i0, i1] -> Stmt_bb54[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb42[1023, 1023] -> Stmt_bb54[0, 0]; Stmt_bb54[1023, 1023] -> Stmt_bb66[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb54[i0, i1] -> Stmt_bb66[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb54[1023, 1023] -> Stmt_bb66[0, 0]; Stmt_bb31[1023, 1023] -> Stmt_bb42[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb31[i0, i1] -> Stmt_bb42[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb31[1023, 1023] -> Stmt_bb42[0, 0]; Stmt_bb162[1023, 1023] -> Stmt_bb174[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb162[i0, i1] -> Stmt_bb174[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb162[1023, 1023] -> Stmt_bb174[0, 0]; Stmt_bb126[1023, 1023] -> Stmt_bb138[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb126[i0, i1] -> Stmt_bb138[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb126[1023, 1023] -> Stmt_bb138[0, 0]; Stmt_bb90[1023, 1023] -> Stmt_bb102[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb90[i0, i1] -> Stmt_bb102[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb90[1023, 1023] -> Stmt_bb102[0, 0]; Stmt_bb138[1023, 1023] -> Stmt_bb150[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb138[i0, i1] -> Stmt_bb150[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb138[1023, 1023] -> Stmt_bb150[0, 0]; Stmt_bb66[1023, 1023] -> Stmt_bb78[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb66[i0, i1] -> Stmt_bb78[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb66[1023, 1023] -> Stmt_bb78[0, 0]; Stmt_bb78[1023, 1023] -> Stmt_bb90[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb78[i0, i1] -> Stmt_bb90[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb78[1023, 1023] -> Stmt_bb90[0, 0]; Stmt_bb114[1023, 1023] -> Stmt_bb126[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb114[i0, i1] -> Stmt_bb126[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb114[1023, 1023] -> Stmt_bb126[0, 0] }
; CHECK-NEXT: 	WAW dependences:
; CHECK-NEXT: 		{ Stmt_bb150[1023, 1023] -> Stmt_bb162[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb150[i0, i1] -> Stmt_bb162[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb150[1023, 1023] -> Stmt_bb162[0, 0]; Stmt_bb174[1023, 1023] -> Stmt_bb186[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb174[i0, i1] -> Stmt_bb186[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb174[1023, 1023] -> Stmt_bb186[0, 0]; Stmt_bb102[1023, 1023] -> Stmt_bb114[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb102[i0, i1] -> Stmt_bb114[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb102[1023, 1023] -> Stmt_bb114[0, 0]; Stmt_bb42[1023, 1023] -> Stmt_bb54[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb42[i0, i1] -> Stmt_bb54[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb42[1023, 1023] -> Stmt_bb54[0, 0]; Stmt_bb54[1023, 1023] -> Stmt_bb66[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb54[i0, i1] -> Stmt_bb66[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb54[1023, 1023] -> Stmt_bb66[0, 0]; Stmt_bb31[1023, 1023] -> Stmt_bb42[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb31[i0, i1] -> Stmt_bb42[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb31[1023, 1023] -> Stmt_bb42[0, 0]; Stmt_bb162[1023, 1023] -> Stmt_bb174[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb162[i0, i1] -> Stmt_bb174[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb162[1023, 1023] -> Stmt_bb174[0, 0]; Stmt_bb126[1023, 1023] -> Stmt_bb138[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb126[i0, i1] -> Stmt_bb138[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb126[1023, 1023] -> Stmt_bb138[0, 0]; Stmt_bb90[1023, 1023] -> Stmt_bb102[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb90[i0, i1] -> Stmt_bb102[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb90[1023, 1023] -> Stmt_bb102[0, 0]; Stmt_bb138[1023, 1023] -> Stmt_bb150[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb138[i0, i1] -> Stmt_bb150[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb138[1023, 1023] -> Stmt_bb150[0, 0]; Stmt_bb66[1023, 1023] -> Stmt_bb78[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb66[i0, i1] -> Stmt_bb78[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb66[1023, 1023] -> Stmt_bb78[0, 0]; Stmt_bb78[1023, 1023] -> Stmt_bb90[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb78[i0, i1] -> Stmt_bb90[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb78[1023, 1023] -> Stmt_bb90[0, 0]; Stmt_bb114[1023, 1023] -> Stmt_bb126[o0, o1] : o0 <= 1023 and o1 >= 0 and -1024o0 < o1 <= 1023; Stmt_bb114[i0, i1] -> Stmt_bb126[0, 0] : i0 >= 0 and 0 <= i1 <= 1048574 - 1024i0 and i1 <= 1023; Stmt_bb114[1023, 1023] -> Stmt_bb126[0, 0] }
; CHECK-NEXT: 	Reduction dependences:
; CHECK-NEXT: 		{ Stmt_bb102[i0, i1] -> Stmt_bb102[i0, 1 + i1] : 0 <= i0 <= 1023 and 0 <= i1 <= 1022; Stmt_bb102[i0, 1023] -> Stmt_bb102[1 + i0, 0] : 0 <= i0 <= 1022; Stmt_bb186[i0, i1] -> Stmt_bb186[i0, 1 + i1] : 0 <= i0 <= 1023 and 0 <= i1 <= 1022; Stmt_bb186[i0, 1023] -> Stmt_bb186[1 + i0, 0] : 0 <= i0 <= 1022; Stmt_bb90[i0, i1] -> Stmt_bb90[i0, 1 + i1] : 0 <= i0 <= 1023 and 0 <= i1 <= 1022; Stmt_bb90[i0, 1023] -> Stmt_bb90[1 + i0, 0] : 0 <= i0 <= 1022; Stmt_bb66[i0, i1] -> Stmt_bb66[i0, 1 + i1] : 0 <= i0 <= 1023 and 0 <= i1 <= 1022; Stmt_bb66[i0, 1023] -> Stmt_bb66[1 + i0, 0] : 0 <= i0 <= 1022; Stmt_bb31[i0, i1] -> Stmt_bb31[i0, 1 + i1] : 0 <= i0 <= 1023 and 0 <= i1 <= 1022; Stmt_bb31[i0, 1023] -> Stmt_bb31[1 + i0, 0] : 0 <= i0 <= 1022; Stmt_bb138[i0, i1] -> Stmt_bb138[i0, 1 + i1] : 0 <= i0 <= 1023 and 0 <= i1 <= 1022; Stmt_bb138[i0, 1023] -> Stmt_bb138[1 + i0, 0] : 0 <= i0 <= 1022; Stmt_bb126[i0, i1] -> Stmt_bb126[i0, 1 + i1] : 0 <= i0 <= 1023 and 0 <= i1 <= 1022; Stmt_bb126[i0, 1023] -> Stmt_bb126[1 + i0, 0] : 0 <= i0 <= 1022; Stmt_bb150[i0, i1] -> Stmt_bb150[i0, 1 + i1] : 0 <= i0 <= 1023 and 0 <= i1 <= 1022; Stmt_bb150[i0, 1023] -> Stmt_bb150[1 + i0, 0] : 0 <= i0 <= 1022; Stmt_bb42[i0, i1] -> Stmt_bb42[i0, 1 + i1] : 0 <= i0 <= 1023 and 0 <= i1 <= 1022; Stmt_bb42[i0, 1023] -> Stmt_bb42[1 + i0, 0] : 0 <= i0 <= 1022; Stmt_bb78[i0, i1] -> Stmt_bb78[i0, 1 + i1] : 0 <= i0 <= 1023 and 0 <= i1 <= 1022; Stmt_bb78[i0, 1023] -> Stmt_bb78[1 + i0, 0] : 0 <= i0 <= 1022; Stmt_bb114[i0, i1] -> Stmt_bb114[i0, 1 + i1] : 0 <= i0 <= 1023 and 0 <= i1 <= 1022; Stmt_bb114[i0, 1023] -> Stmt_bb114[1 + i0, 0] : 0 <= i0 <= 1022; Stmt_bb174[i0, i1] -> Stmt_bb174[i0, 1 + i1] : 0 <= i0 <= 1023 and 0 <= i1 <= 1022; Stmt_bb174[i0, 1023] -> Stmt_bb174[1 + i0, 0] : 0 <= i0 <= 1022; Stmt_bb162[i0, i1] -> Stmt_bb162[i0, 1 + i1] : 0 <= i0 <= 1023 and 0 <= i1 <= 1022; Stmt_bb162[i0, 1023] -> Stmt_bb162[1 + i0, 0] : 0 <= i0 <= 1022; Stmt_bb54[i0, i1] -> Stmt_bb54[i0, 1 + i1] : 0 <= i0 <= 1023 and 0 <= i1 <= 1022; Stmt_bb54[i0, 1023] -> Stmt_bb54[1 + i0, 0] : 0 <= i0 <= 1022 }
; CHECK-NEXT: 	Transitive closure of reduction dependences:
; CHECK-NEXT: 		{ Stmt_bb102[i0, i1] -> Stmt_bb102[o0, o1] : 0 <= i1 <= 1023 and 0 <= o1 <= 1023 and ((i0 >= 0 and o0 <= 1023 and o1 > 1024i0 + i1 - 1024o0) or (i0 <= 1023 and o0 >= 0 and o1 < 1024i0 + i1 - 1024o0)); Stmt_bb186[i0, i1] -> Stmt_bb186[o0, o1] : 0 <= i1 <= 1023 and 0 <= o1 <= 1023 and ((i0 >= 0 and o0 <= 1023 and o1 > 1024i0 + i1 - 1024o0) or (i0 <= 1023 and o0 >= 0 and o1 < 1024i0 + i1 - 1024o0)); Stmt_bb90[i0, i1] -> Stmt_bb90[o0, o1] : 0 <= i1 <= 1023 and 0 <= o1 <= 1023 and ((i0 >= 0 and o0 <= 1023 and o1 > 1024i0 + i1 - 1024o0) or (i0 <= 1023 and o0 >= 0 and o1 < 1024i0 + i1 - 1024o0)); Stmt_bb66[i0, i1] -> Stmt_bb66[o0, o1] : 0 <= i1 <= 1023 and 0 <= o1 <= 1023 and ((i0 >= 0 and o0 <= 1023 and o1 > 1024i0 + i1 - 1024o0) or (i0 <= 1023 and o0 >= 0 and o1 < 1024i0 + i1 - 1024o0)); Stmt_bb31[i0, i1] -> Stmt_bb31[o0, o1] : 0 <= i1 <= 1023 and 0 <= o1 <= 1023 and ((i0 >= 0 and o0 <= 1023 and o1 > 1024i0 + i1 - 1024o0) or (i0 <= 1023 and o0 >= 0 and o1 < 1024i0 + i1 - 1024o0)); Stmt_bb138[i0, i1] -> Stmt_bb138[o0, o1] : 0 <= i1 <= 1023 and 0 <= o1 <= 1023 and ((i0 >= 0 and o0 <= 1023 and o1 > 1024i0 + i1 - 1024o0) or (i0 <= 1023 and o0 >= 0 and o1 < 1024i0 + i1 - 1024o0)); Stmt_bb126[i0, i1] -> Stmt_bb126[o0, o1] : 0 <= i1 <= 1023 and 0 <= o1 <= 1023 and ((i0 >= 0 and o0 <= 1023 and o1 > 1024i0 + i1 - 1024o0) or (i0 <= 1023 and o0 >= 0 and o1 < 1024i0 + i1 - 1024o0)); Stmt_bb150[i0, i1] -> Stmt_bb150[o0, o1] : 0 <= i1 <= 1023 and 0 <= o1 <= 1023 and ((i0 >= 0 and o0 <= 1023 and o1 > 1024i0 + i1 - 1024o0) or (i0 <= 1023 and o0 >= 0 and o1 < 1024i0 + i1 - 1024o0)); Stmt_bb42[i0, i1] -> Stmt_bb42[o0, o1] : 0 <= i1 <= 1023 and 0 <= o1 <= 1023 and ((i0 >= 0 and o0 <= 1023 and o1 > 1024i0 + i1 - 1024o0) or (i0 <= 1023 and o0 >= 0 and o1 < 1024i0 + i1 - 1024o0)); Stmt_bb78[i0, i1] -> Stmt_bb78[o0, o1] : 0 <= i1 <= 1023 and 0 <= o1 <= 1023 and ((i0 >= 0 and o0 <= 1023 and o1 > 1024i0 + i1 - 1024o0) or (i0 <= 1023 and o0 >= 0 and o1 < 1024i0 + i1 - 1024o0)); Stmt_bb114[i0, i1] -> Stmt_bb114[o0, o1] : 0 <= i1 <= 1023 and 0 <= o1 <= 1023 and ((i0 >= 0 and o0 <= 1023 and o1 > 1024i0 + i1 - 1024o0) or (i0 <= 1023 and o0 >= 0 and o1 < 1024i0 + i1 - 1024o0)); Stmt_bb174[i0, i1] -> Stmt_bb174[o0, o1] : 0 <= i1 <= 1023 and 0 <= o1 <= 1023 and ((i0 >= 0 and o0 <= 1023 and o1 > 1024i0 + i1 - 1024o0) or (i0 <= 1023 and o0 >= 0 and o1 < 1024i0 + i1 - 1024o0)); Stmt_bb162[i0, i1] -> Stmt_bb162[o0, o1] : 0 <= i1 <= 1023 and 0 <= o1 <= 1023 and ((i0 >= 0 and o0 <= 1023 and o1 > 1024i0 + i1 - 1024o0) or (i0 <= 1023 and o0 >= 0 and o1 < 1024i0 + i1 - 1024o0)); Stmt_bb54[i0, i1] -> Stmt_bb54[o0, o1] : 0 <= i1 <= 1023 and 0 <= o1 <= 1023 and ((i0 >= 0 and o0 <= 1023 and o1 > 1024i0 + i1 - 1024o0) or (i0 <= 1023 and o0 >= 0 and o1 < 1024i0 + i1 - 1024o0)) }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @manyreductions(i64* %A) {
bb:
  br label %bb28

bb28:                                             ; preds = %bb36, %bb
  %i.0 = phi i64 [ 0, %bb ], [ %tmp37, %bb36 ]
  %exitcond27 = icmp ne i64 %i.0, 1024
  br i1 %exitcond27, label %bb29, label %bb38

bb29:                                             ; preds = %bb28
  br label %bb30

bb30:                                             ; preds = %bb33, %bb29
  %j.0 = phi i64 [ 0, %bb29 ], [ %tmp34, %bb33 ]
  %exitcond26 = icmp ne i64 %j.0, 1024
  br i1 %exitcond26, label %bb31, label %bb35

bb31:                                             ; preds = %bb30
  %tmp = load i64, i64* %A, align 8
  %tmp32 = add nsw i64 %tmp, 42
  store i64 %tmp32, i64* %A, align 8
  br label %bb33

bb33:                                             ; preds = %bb31
  %tmp34 = add nuw nsw i64 %j.0, 1
  br label %bb30

bb35:                                             ; preds = %bb30
  br label %bb36

bb36:                                             ; preds = %bb35
  %tmp37 = add nuw nsw i64 %i.0, 1
  br label %bb28

bb38:                                             ; preds = %bb28
  br label %bb39

bb39:                                             ; preds = %bb48, %bb38
  %i1.0 = phi i64 [ 0, %bb38 ], [ %tmp49, %bb48 ]
  %exitcond25 = icmp ne i64 %i1.0, 1024
  br i1 %exitcond25, label %bb40, label %bb50

bb40:                                             ; preds = %bb39
  br label %bb41

bb41:                                             ; preds = %bb45, %bb40
  %j2.0 = phi i64 [ 0, %bb40 ], [ %tmp46, %bb45 ]
  %exitcond24 = icmp ne i64 %j2.0, 1024
  br i1 %exitcond24, label %bb42, label %bb47

bb42:                                             ; preds = %bb41
  %tmp43 = load i64, i64* %A, align 8
  %tmp44 = add nsw i64 %tmp43, 42
  store i64 %tmp44, i64* %A, align 8
  br label %bb45

bb45:                                             ; preds = %bb42
  %tmp46 = add nuw nsw i64 %j2.0, 1
  br label %bb41

bb47:                                             ; preds = %bb41
  br label %bb48

bb48:                                             ; preds = %bb47
  %tmp49 = add nuw nsw i64 %i1.0, 1
  br label %bb39

bb50:                                             ; preds = %bb39
  br label %bb51

bb51:                                             ; preds = %bb60, %bb50
  %i3.0 = phi i64 [ 0, %bb50 ], [ %tmp61, %bb60 ]
  %exitcond23 = icmp ne i64 %i3.0, 1024
  br i1 %exitcond23, label %bb52, label %bb62

bb52:                                             ; preds = %bb51
  br label %bb53

bb53:                                             ; preds = %bb57, %bb52
  %j4.0 = phi i64 [ 0, %bb52 ], [ %tmp58, %bb57 ]
  %exitcond22 = icmp ne i64 %j4.0, 1024
  br i1 %exitcond22, label %bb54, label %bb59

bb54:                                             ; preds = %bb53
  %tmp55 = load i64, i64* %A, align 8
  %tmp56 = add nsw i64 %tmp55, 42
  store i64 %tmp56, i64* %A, align 8
  br label %bb57

bb57:                                             ; preds = %bb54
  %tmp58 = add nuw nsw i64 %j4.0, 1
  br label %bb53

bb59:                                             ; preds = %bb53
  br label %bb60

bb60:                                             ; preds = %bb59
  %tmp61 = add nuw nsw i64 %i3.0, 1
  br label %bb51

bb62:                                             ; preds = %bb51
  br label %bb63

bb63:                                             ; preds = %bb72, %bb62
  %i5.0 = phi i64 [ 0, %bb62 ], [ %tmp73, %bb72 ]
  %exitcond21 = icmp ne i64 %i5.0, 1024
  br i1 %exitcond21, label %bb64, label %bb74

bb64:                                             ; preds = %bb63
  br label %bb65

bb65:                                             ; preds = %bb69, %bb64
  %j6.0 = phi i64 [ 0, %bb64 ], [ %tmp70, %bb69 ]
  %exitcond20 = icmp ne i64 %j6.0, 1024
  br i1 %exitcond20, label %bb66, label %bb71

bb66:                                             ; preds = %bb65
  %tmp67 = load i64, i64* %A, align 8
  %tmp68 = add nsw i64 %tmp67, 42
  store i64 %tmp68, i64* %A, align 8
  br label %bb69

bb69:                                             ; preds = %bb66
  %tmp70 = add nuw nsw i64 %j6.0, 1
  br label %bb65

bb71:                                             ; preds = %bb65
  br label %bb72

bb72:                                             ; preds = %bb71
  %tmp73 = add nuw nsw i64 %i5.0, 1
  br label %bb63

bb74:                                             ; preds = %bb63
  br label %bb75

bb75:                                             ; preds = %bb84, %bb74
  %i7.0 = phi i64 [ 0, %bb74 ], [ %tmp85, %bb84 ]
  %exitcond19 = icmp ne i64 %i7.0, 1024
  br i1 %exitcond19, label %bb76, label %bb86

bb76:                                             ; preds = %bb75
  br label %bb77

bb77:                                             ; preds = %bb81, %bb76
  %j8.0 = phi i64 [ 0, %bb76 ], [ %tmp82, %bb81 ]
  %exitcond18 = icmp ne i64 %j8.0, 1024
  br i1 %exitcond18, label %bb78, label %bb83

bb78:                                             ; preds = %bb77
  %tmp79 = load i64, i64* %A, align 8
  %tmp80 = add nsw i64 %tmp79, 42
  store i64 %tmp80, i64* %A, align 8
  br label %bb81

bb81:                                             ; preds = %bb78
  %tmp82 = add nuw nsw i64 %j8.0, 1
  br label %bb77

bb83:                                             ; preds = %bb77
  br label %bb84

bb84:                                             ; preds = %bb83
  %tmp85 = add nuw nsw i64 %i7.0, 1
  br label %bb75

bb86:                                             ; preds = %bb75
  br label %bb87

bb87:                                             ; preds = %bb96, %bb86
  %i9.0 = phi i64 [ 0, %bb86 ], [ %tmp97, %bb96 ]
  %exitcond17 = icmp ne i64 %i9.0, 1024
  br i1 %exitcond17, label %bb88, label %bb98

bb88:                                             ; preds = %bb87
  br label %bb89

bb89:                                             ; preds = %bb93, %bb88
  %j10.0 = phi i64 [ 0, %bb88 ], [ %tmp94, %bb93 ]
  %exitcond16 = icmp ne i64 %j10.0, 1024
  br i1 %exitcond16, label %bb90, label %bb95

bb90:                                             ; preds = %bb89
  %tmp91 = load i64, i64* %A, align 8
  %tmp92 = add nsw i64 %tmp91, 42
  store i64 %tmp92, i64* %A, align 8
  br label %bb93

bb93:                                             ; preds = %bb90
  %tmp94 = add nuw nsw i64 %j10.0, 1
  br label %bb89

bb95:                                             ; preds = %bb89
  br label %bb96

bb96:                                             ; preds = %bb95
  %tmp97 = add nuw nsw i64 %i9.0, 1
  br label %bb87

bb98:                                             ; preds = %bb87
  br label %bb99

bb99:                                             ; preds = %bb108, %bb98
  %i11.0 = phi i64 [ 0, %bb98 ], [ %tmp109, %bb108 ]
  %exitcond15 = icmp ne i64 %i11.0, 1024
  br i1 %exitcond15, label %bb100, label %bb110

bb100:                                            ; preds = %bb99
  br label %bb101

bb101:                                            ; preds = %bb105, %bb100
  %j12.0 = phi i64 [ 0, %bb100 ], [ %tmp106, %bb105 ]
  %exitcond14 = icmp ne i64 %j12.0, 1024
  br i1 %exitcond14, label %bb102, label %bb107

bb102:                                            ; preds = %bb101
  %tmp103 = load i64, i64* %A, align 8
  %tmp104 = add nsw i64 %tmp103, 42
  store i64 %tmp104, i64* %A, align 8
  br label %bb105

bb105:                                            ; preds = %bb102
  %tmp106 = add nuw nsw i64 %j12.0, 1
  br label %bb101

bb107:                                            ; preds = %bb101
  br label %bb108

bb108:                                            ; preds = %bb107
  %tmp109 = add nuw nsw i64 %i11.0, 1
  br label %bb99

bb110:                                            ; preds = %bb99
  br label %bb111

bb111:                                            ; preds = %bb120, %bb110
  %i13.0 = phi i64 [ 0, %bb110 ], [ %tmp121, %bb120 ]
  %exitcond13 = icmp ne i64 %i13.0, 1024
  br i1 %exitcond13, label %bb112, label %bb122

bb112:                                            ; preds = %bb111
  br label %bb113

bb113:                                            ; preds = %bb117, %bb112
  %j14.0 = phi i64 [ 0, %bb112 ], [ %tmp118, %bb117 ]
  %exitcond12 = icmp ne i64 %j14.0, 1024
  br i1 %exitcond12, label %bb114, label %bb119

bb114:                                            ; preds = %bb113
  %tmp115 = load i64, i64* %A, align 8
  %tmp116 = add nsw i64 %tmp115, 42
  store i64 %tmp116, i64* %A, align 8
  br label %bb117

bb117:                                            ; preds = %bb114
  %tmp118 = add nuw nsw i64 %j14.0, 1
  br label %bb113

bb119:                                            ; preds = %bb113
  br label %bb120

bb120:                                            ; preds = %bb119
  %tmp121 = add nuw nsw i64 %i13.0, 1
  br label %bb111

bb122:                                            ; preds = %bb111
  br label %bb123

bb123:                                            ; preds = %bb132, %bb122
  %i15.0 = phi i64 [ 0, %bb122 ], [ %tmp133, %bb132 ]
  %exitcond11 = icmp ne i64 %i15.0, 1024
  br i1 %exitcond11, label %bb124, label %bb134

bb124:                                            ; preds = %bb123
  br label %bb125

bb125:                                            ; preds = %bb129, %bb124
  %j16.0 = phi i64 [ 0, %bb124 ], [ %tmp130, %bb129 ]
  %exitcond10 = icmp ne i64 %j16.0, 1024
  br i1 %exitcond10, label %bb126, label %bb131

bb126:                                            ; preds = %bb125
  %tmp127 = load i64, i64* %A, align 8
  %tmp128 = add nsw i64 %tmp127, 42
  store i64 %tmp128, i64* %A, align 8
  br label %bb129

bb129:                                            ; preds = %bb126
  %tmp130 = add nuw nsw i64 %j16.0, 1
  br label %bb125

bb131:                                            ; preds = %bb125
  br label %bb132

bb132:                                            ; preds = %bb131
  %tmp133 = add nuw nsw i64 %i15.0, 1
  br label %bb123

bb134:                                            ; preds = %bb123
  br label %bb135

bb135:                                            ; preds = %bb144, %bb134
  %i17.0 = phi i64 [ 0, %bb134 ], [ %tmp145, %bb144 ]
  %exitcond9 = icmp ne i64 %i17.0, 1024
  br i1 %exitcond9, label %bb136, label %bb146

bb136:                                            ; preds = %bb135
  br label %bb137

bb137:                                            ; preds = %bb141, %bb136
  %j18.0 = phi i64 [ 0, %bb136 ], [ %tmp142, %bb141 ]
  %exitcond8 = icmp ne i64 %j18.0, 1024
  br i1 %exitcond8, label %bb138, label %bb143

bb138:                                            ; preds = %bb137
  %tmp139 = load i64, i64* %A, align 8
  %tmp140 = add nsw i64 %tmp139, 42
  store i64 %tmp140, i64* %A, align 8
  br label %bb141

bb141:                                            ; preds = %bb138
  %tmp142 = add nuw nsw i64 %j18.0, 1
  br label %bb137

bb143:                                            ; preds = %bb137
  br label %bb144

bb144:                                            ; preds = %bb143
  %tmp145 = add nuw nsw i64 %i17.0, 1
  br label %bb135

bb146:                                            ; preds = %bb135
  br label %bb147

bb147:                                            ; preds = %bb156, %bb146
  %i19.0 = phi i64 [ 0, %bb146 ], [ %tmp157, %bb156 ]
  %exitcond7 = icmp ne i64 %i19.0, 1024
  br i1 %exitcond7, label %bb148, label %bb158

bb148:                                            ; preds = %bb147
  br label %bb149

bb149:                                            ; preds = %bb153, %bb148
  %j20.0 = phi i64 [ 0, %bb148 ], [ %tmp154, %bb153 ]
  %exitcond6 = icmp ne i64 %j20.0, 1024
  br i1 %exitcond6, label %bb150, label %bb155

bb150:                                            ; preds = %bb149
  %tmp151 = load i64, i64* %A, align 8
  %tmp152 = add nsw i64 %tmp151, 42
  store i64 %tmp152, i64* %A, align 8
  br label %bb153

bb153:                                            ; preds = %bb150
  %tmp154 = add nuw nsw i64 %j20.0, 1
  br label %bb149

bb155:                                            ; preds = %bb149
  br label %bb156

bb156:                                            ; preds = %bb155
  %tmp157 = add nuw nsw i64 %i19.0, 1
  br label %bb147

bb158:                                            ; preds = %bb147
  br label %bb159

bb159:                                            ; preds = %bb168, %bb158
  %i21.0 = phi i64 [ 0, %bb158 ], [ %tmp169, %bb168 ]
  %exitcond5 = icmp ne i64 %i21.0, 1024
  br i1 %exitcond5, label %bb160, label %bb170

bb160:                                            ; preds = %bb159
  br label %bb161

bb161:                                            ; preds = %bb165, %bb160
  %j22.0 = phi i64 [ 0, %bb160 ], [ %tmp166, %bb165 ]
  %exitcond4 = icmp ne i64 %j22.0, 1024
  br i1 %exitcond4, label %bb162, label %bb167

bb162:                                            ; preds = %bb161
  %tmp163 = load i64, i64* %A, align 8
  %tmp164 = add nsw i64 %tmp163, 42
  store i64 %tmp164, i64* %A, align 8
  br label %bb165

bb165:                                            ; preds = %bb162
  %tmp166 = add nuw nsw i64 %j22.0, 1
  br label %bb161

bb167:                                            ; preds = %bb161
  br label %bb168

bb168:                                            ; preds = %bb167
  %tmp169 = add nuw nsw i64 %i21.0, 1
  br label %bb159

bb170:                                            ; preds = %bb159
  br label %bb171

bb171:                                            ; preds = %bb180, %bb170
  %i23.0 = phi i64 [ 0, %bb170 ], [ %tmp181, %bb180 ]
  %exitcond3 = icmp ne i64 %i23.0, 1024
  br i1 %exitcond3, label %bb172, label %bb182

bb172:                                            ; preds = %bb171
  br label %bb173

bb173:                                            ; preds = %bb177, %bb172
  %j24.0 = phi i64 [ 0, %bb172 ], [ %tmp178, %bb177 ]
  %exitcond2 = icmp ne i64 %j24.0, 1024
  br i1 %exitcond2, label %bb174, label %bb179

bb174:                                            ; preds = %bb173
  %tmp175 = load i64, i64* %A, align 8
  %tmp176 = add nsw i64 %tmp175, 42
  store i64 %tmp176, i64* %A, align 8
  br label %bb177

bb177:                                            ; preds = %bb174
  %tmp178 = add nuw nsw i64 %j24.0, 1
  br label %bb173

bb179:                                            ; preds = %bb173
  br label %bb180

bb180:                                            ; preds = %bb179
  %tmp181 = add nuw nsw i64 %i23.0, 1
  br label %bb171

bb182:                                            ; preds = %bb171
  br label %bb183

bb183:                                            ; preds = %bb192, %bb182
  %i25.0 = phi i64 [ 0, %bb182 ], [ %tmp193, %bb192 ]
  %exitcond1 = icmp ne i64 %i25.0, 1024
  br i1 %exitcond1, label %bb184, label %bb194

bb184:                                            ; preds = %bb183
  br label %bb185

bb185:                                            ; preds = %bb189, %bb184
  %j26.0 = phi i64 [ 0, %bb184 ], [ %tmp190, %bb189 ]
  %exitcond = icmp ne i64 %j26.0, 1024
  br i1 %exitcond, label %bb186, label %bb191

bb186:                                            ; preds = %bb185
  %tmp187 = load i64, i64* %A, align 8
  %tmp188 = add nsw i64 %tmp187, 42
  store i64 %tmp188, i64* %A, align 8
  br label %bb189

bb189:                                            ; preds = %bb186
  %tmp190 = add nuw nsw i64 %j26.0, 1
  br label %bb185

bb191:                                            ; preds = %bb185
  br label %bb192

bb192:                                            ; preds = %bb191
  %tmp193 = add nuw nsw i64 %i25.0, 1
  br label %bb183

bb194:                                            ; preds = %bb183
  ret void
}
