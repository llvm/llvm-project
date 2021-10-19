; Checks if llc can deal with different char encodings.
; This is only required for z/OS.
;
; UNSUPPORTED: !s390x-none-zos
;
; RUN: cat %s >%t && chtag -tc ISO8859-1 %t && llc %t -o - >/dev/null
; RUN: iconv -f ISO8859-1 -t IBM-1047 <%s >%t && chtag -tc IBM-1047 %t && llc %t -o - >/dev/null
; RUN: iconv -f ISO8859-1 -t IBM-1047 <%s >%t && chtag -r %t && llc %t -o - >/dev/null

@g_105 = external dso_local global i8, align 2
