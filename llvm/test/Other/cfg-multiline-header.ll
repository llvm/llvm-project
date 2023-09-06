; RUN: opt < %s -passes=dot-cfg -cfg-dot-filename-prefix=cfg 2>/dev/null > /dev/null
; RUN: FileCheck %s -input-file=cfg.foo.dot --check-prefix=CHECK

define void @foo(ptr %A, ptr %B) {
a_very_long_label_that_should_take_over_eight_symbols_and_span_2_lines_in_cfg_dot_graph:
; CHECK: label="{a_very_long_label_that_should_take_over_eight_symbols_and_span_2_lines_in_cfg_do\l...t_graph:\l|  store i32 1, ptr %A, align 4\l  store i32 2, ptr %B, align 4\l  br label %short_label\l}"
  store i32 1, ptr %A
  store i32 2, ptr %B
  br label %short_label
short_label:
; CHECK: label="{short_label:\l|  br label\l... %an_even_longer_multiline_label_that_will_span_muliple_lines_Lorem_ipsum_dolor\l..._sit_amet_consectetur_adipiscing_elit_sed_do_eiusmod_tempor_incididunt_ut_labor\l...e_et_dolore_magna_aliqua\l}"
  br label %an_even_longer_multiline_label_that_will_span_muliple_lines_Lorem_ipsum_dolor_sit_amet_consectetur_adipiscing_elit_sed_do_eiusmod_tempor_incididunt_ut_labore_et_dolore_magna_aliqua
an_even_longer_multiline_label_that_will_span_muliple_lines_Lorem_ipsum_dolor_sit_amet_consectetur_adipiscing_elit_sed_do_eiusmod_tempor_incididunt_ut_labore_et_dolore_magna_aliqua:
; CHECK: label="{an_even_longer_multiline_label_that_will_span_muliple_lines_Lorem_ipsum_dolor_si\l...t_amet_consectetur_adipiscing_elit_sed_do_eiusmod_tempor_incididunt_ut_labore_e\l...t_dolore_magna_aliqua:\l|  ret void\l}"  
  ret void
}
