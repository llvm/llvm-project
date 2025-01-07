; RUN: llc -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr8 \
; RUN:   -verify-machineinstrs -stop-after=finalize-isel  < %s | \
; RUN:   FileCheck --check-prefixes=POWR8,VSX %s
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr7 \
; RUN:   -verify-machineinstrs -mattr=-vsx -stop-after=finalize-isel \
; RUN:    < %s | FileCheck %s --check-prefixes=NOVSX,NOP8V
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr7 \
; RUN:   -verify-machineinstrs -mattr=vsx -stop-after=finalize-isel \
; RUN:    < %s | FileCheck %s --check-prefixes=VSX,NOP8V

define float @vssr(float %a, float %b, float %c, float %d, float %e) {
entry:
  %add = fadd float %a, %b
  %add1 = fadd float %add, %c
  %add2 = fadd float %add1, %d
  %add3 = fadd float %add2, %e
  ret float %add3
}

; POWR8-LABEL: name:            vssr
; POWR8:       - { id: 0, class: vssrc, preferred-register: '', flags: [  ] }
; POWR8:       - { id: 1, class: vssrc, preferred-register: '', flags: [  ] }
; POWR8:       - { id: 2, class: vssrc, preferred-register: '', flags: [  ] }
; POWR8:       - { id: 3, class: vssrc, preferred-register: '', flags: [  ] }
; POWR8:       - { id: 4, class: vssrc, preferred-register: '', flags: [  ] }
; POWR8:       - { id: 5, class: vssrc, preferred-register: '', flags: [  ] }
; POWR8:       - { id: 6, class: vssrc, preferred-register: '', flags: [  ] }
; POWR8:       - { id: 7, class: vssrc, preferred-register: '', flags: [  ] }
; POWR8:       - { id: 8, class: vssrc, preferred-register: '', flags: [  ] }
; POWR8:       %4:vssrc = COPY $f5
; POWR8:       %3:vssrc = COPY $f4
; POWR8:       %2:vssrc = COPY $f3
; POWR8:       %1:vssrc = COPY $f2
; POWR8:       %0:vssrc = COPY $f1
; POWR8:       %5:vssrc = nofpexcept XSADDSP %0, %1
; POWR8:       %6:vssrc = nofpexcept XSADDSP killed %5, %2
; POWR8:       %7:vssrc = nofpexcept XSADDSP killed %6, %3
; POWR8:       %8:vssrc = nofpexcept XSADDSP killed %7, %4
; POWR8:       $f1 = COPY %8

; NOP8V-LABEL: name:            vssr
; NOP8V:       registers:
; NOP8V:       - { id: 0, class: f4rc, preferred-register: '', flags: [  ] }
; NOP8V:       - { id: 1, class: f4rc, preferred-register: '', flags: [  ] }
; NOP8V:       - { id: 2, class: f4rc, preferred-register: '', flags: [  ] }
; NOP8V:       - { id: 3, class: f4rc, preferred-register: '', flags: [  ] }
; NOP8V:       - { id: 4, class: f4rc, preferred-register: '', flags: [  ] }
; NOP8V:       - { id: 5, class: f4rc, preferred-register: '', flags: [  ] }
; NOP8V:       - { id: 6, class: f4rc, preferred-register: '', flags: [  ] }
; NOP8V:       - { id: 7, class: f4rc, preferred-register: '', flags: [  ] }
; NOP8V:       - { id: 8, class: f4rc, preferred-register: '', flags: [  ] }
; NOP8V:        %4:f4rc = COPY $f5
; NOP8V:        %3:f4rc = COPY $f4
; NOP8V:        %2:f4rc = COPY $f3
; NOP8V:        %1:f4rc = COPY $f2
; NOP8V:        %0:f4rc = COPY $f1
; NOP8V:        %5:f4rc = nofpexcept FADDS %0, %1, implicit $rm
; NOP8V:        %6:f4rc = nofpexcept FADDS killed %5, %2, implicit $rm
; NOP8V:        %7:f4rc = nofpexcept FADDS killed %6, %3, implicit $rm
; NOP8V:        %8:f4rc = nofpexcept FADDS killed %7, %4, implicit $rm
; NOP8V:        $f1 = COPY %8

define double @vsfr(double %a, double %b, double %c, double %d, double %e) {
entry:
  %add = fadd double %a, %b
  %add1 = fadd double %add, %c
  %add2 = fadd double %add1, %d
  %add3 = fadd double %add2, %e
  ret double %add3
}

; VSX-LABEL:   vsfr
; VSX:         registers:
; VSX:          - { id: 0, class: vsfrc, preferred-register: '', flags: [  ] }
; VSX:          - { id: 1, class: vsfrc, preferred-register: '', flags: [  ] }
; VSX:          - { id: 2, class: vsfrc, preferred-register: '', flags: [  ] }
; VSX:          - { id: 3, class: vsfrc, preferred-register: '', flags: [  ] }
; VSX:          - { id: 4, class: vsfrc, preferred-register: '', flags: [  ] }
; VSX:          - { id: 5, class: vsfrc, preferred-register: '', flags: [  ] }
; VSX:          - { id: 6, class: vsfrc, preferred-register: '', flags: [  ] }
; VSX:          - { id: 7, class: vsfrc, preferred-register: '', flags: [  ] }
; VSX:          - { id: 8, class: vsfrc, preferred-register: '', flags: [  ] }
; VSX:          %4:vsfrc = COPY $f5
; VSX:          %3:vsfrc = COPY $f4
; VSX:          %2:vsfrc = COPY $f3
; VSX:          %1:vsfrc = COPY $f2
; VSX:          %0:vsfrc = COPY $f1
; VSX:          %5:vsfrc = nofpexcept XSADDDP %0, %1, implicit $rm
; VSX:          %6:vsfrc = nofpexcept XSADDDP killed %5, %2, implicit $rm
; VSX:          %7:vsfrc = nofpexcept XSADDDP killed %6, %3, implicit $rm
; VSX:          %8:vsfrc = nofpexcept XSADDDP killed %7, %4, implicit $rm
; VSX:          $f1 = COPY %8

; NOVSX-LABEL:  vsfr
; NOVSX:        registers:
; NOVSX:        - { id: 0, class: f8rc, preferred-register: '', flags: [  ] }
; NOVSX:        - { id: 1, class: f8rc, preferred-register: '', flags: [  ] }
; NOVSX:        - { id: 2, class: f8rc, preferred-register: '', flags: [  ] }
; NOVSX:        - { id: 3, class: f8rc, preferred-register: '', flags: [  ] }
; NOVSX:        - { id: 4, class: f8rc, preferred-register: '', flags: [  ] }
; NOVSX:        - { id: 5, class: f8rc, preferred-register: '', flags: [  ] }
; NOVSX:        - { id: 6, class: f8rc, preferred-register: '', flags: [  ] }
; NOVSX:        - { id: 7, class: f8rc, preferred-register: '', flags: [  ] }
; NOVSX:        - { id: 8, class: f8rc, preferred-register: '', flags: [  ] }
; NOVSX:        %4:f8rc = COPY $f5
; NOVSX:        %3:f8rc = COPY $f4
; NOVSX:        %2:f8rc = COPY $f3
; NOVSX:        %1:f8rc = COPY $f2
; NOVSX:        %0:f8rc = COPY $f1
; NOVSX:        %5:f8rc = nofpexcept FADD %0, %1, implicit $rm
; NOVSX:        %6:f8rc = nofpexcept FADD killed %5, %2, implicit $rm
; NOVSX:        %7:f8rc = nofpexcept FADD killed %6, %3, implicit $rm
; NOVSX:        %8:f8rc = nofpexcept FADD killed %7, %4, implicit $rm
; NOVSX:        $f1 = COPY %8

