# RUN: llvm-mc -triple=hexagon -filetype=obj %s | llvm-objdump --no-print-imm-hex -d - | FileCheck %s
#

# Currently ignore if there is one or two #'s

		r7 = memw(gp+#192)
# CHECK:	r7 = memw(gp+#192)

		r3:2 = memd(gp+#64)
# CHECK: 	r3:2 = memd(gp+#64)

		{ p3 = p1; r8 = #2; if (p3.new) memw(##8) = r8.new }
# CHECK:	if (p3.new) memw({{..}}8) = r8

		{ p3 = p1; r8 = #2; if (!p3.new) memw(##8) = r8.new }
# CHECK:	if (!p3.new) memw({{..}}8) = r8.new

		{ r8 = #2; if (p3) memw(##8) = r8.new }
# CHECK:	if (p3) memw({{..}}8) = r8.new

		{ r8 = #2; if (!p3) memw(##8) = r8.new }
# CHECK:	if (!p3) memw({{..}}8) = r8.new

		{ p3 = p1; r8 = #2; if (p3.new) memh(##8) = r8.new }
# CHECK:	if (p3.new) memh({{..}}8) = r8.new

		{ p3 = p1; r8 = #2; if (!p3.new) memh(##8) = r8.new }
# CHECK:	if (!p3.new) memh({{..}}8) = r8.new

		{ r8 = #2; if (p3) memh(##8) = r8.new }
# CHECK:	memh({{..}}8) = r8.new

		{ r8 = #2; if (!p3) memh(##8) = r8.new }
# CHECK:	if (!p3) memh({{..}}8) = r8.new

		{ p3 = p1; r8 = #2; if (p3.new) memb(##8) = r8.new }
# CHECK:	if (p3.new) memb({{..}}8) = r8.new

		{ p3 = p1; r8 = #2; if (!p3.new) memb(##8) = r8.new }
# CHECK:	if (!p3.new) memb({{..}}8) = r8.new

		{ r8 = #2; if (p3) memb(##8) = r8.new }
# CHECK:	if (p3) memb({{..}}8) = r8.new

		{ r8 = #2; if (!p3) memb(##8) = r8.new }
# CHECK:	if (!p3) memb({{..}}8) = r8.new

		{ if (p3) memw(##8) = r8 }
# CHECK:	if (p3) memw({{..}}8) = r8

		{ if (!p3) memw(##8) = r8 }
# CHECK:	if (!p3) memw({{..}}8) = r8

		{ p3 = p1; if (p3.new) memw(##8) = r8 }
# CHECK:	if (p3.new) memw({{..}}8) = r8

		{ p3 = p1; if (!p3.new) memw(##8) = r8 }
# CHECK:	if (!p3.new) memw({{..}}8) = r8


		if (!p2) r14 = memb(##48)
# CHECK:	if (!p2) r14 = memb({{..}}48)

		if (p2) r14 = memb(##48)
# CHECK:	if (p2) r14 = memb({{..}}48)

		{p2 = p0; if (!p2.new) r14 = memb(##48) }
# CHECK:	if (!p2.new) r14 = memb({{..}}48)

		{p3 = p2; if (p3.new) r14 = memb(##48) }
# CHECK:	if (p3.new) r14 = memb({{..}}48)


		if (!p2) r14 = memh(##48)
# CHECK:	if (!p2) r14 = memh({{..}}48)

		if (p2) r14 = memh(##48)
# CHECK:	if (p2) r14 = memh({{..}}48)

		{p2 = p0; if (!p2.new) r14 = memh(##48) }
# CHECK:	if (!p2.new) r14 = memh({{..}}48)

		{p3 = p2; if (p3.new) r14 = memh(##48) }
# CHECK:	if (p3.new) r14 = memh({{..}}48)


		if (!p2) r14 = memub(##48)
# CHECK:	if (!p2) r14 = memub({{..}}48)

		if (p2) r14 = memub(##48)
# CHECK:	if (p2) r14 = memub({{..}}48)

		{p2 = p0; if (!p2.new) r14 = memub(##48) }
# CHECK:	if (!p2.new) r14 = memub({{..}}48)

		{p3 = p2; if (p3.new) r14 = memub(##48) }
# CHECK:	if (p3.new) r14 = memub({{..}}48)


		if (!p2) r14 = memuh(##48)
# CHECK:	if (!p2) r14 = memuh({{..}}48)

		if (p2) r14 = memuh(##48)
# CHECK:	if (p2) r14 = memuh({{..}}48)

		{p2 = p0; if (!p2.new) r14 = memuh(##48) }
# CHECK:	if (!p2.new) r14 = memuh({{..}}48)

		{p3 = p2; if (p3.new) r14 = memuh(##48) }
# CHECK:	r14 = memuh({{..}}48)


		if (!p2) r14 = memw(##48)
# CHECK:	if (!p2) r14 = memw({{..}}48)

		if (p2) r14 = memw(##48)
# CHECK:	if (p2) r14 = memw({{..}}48)

		{p2 = p0; if (!p2.new) r14 = memw(##48) }
# CHECK:	if (!p2.new) r14 = memw({{..}}48)

		{p3 = p2; if (p3.new) r14 = memw(##48) }
# CHECK:	if (p3.new) r14 = memw({{..}}48)

		r7 = memh(##32)
# CHECK: 	r7 = memh(##32)
		r7 = memuh(##32)
# CHECK: 	r7 = memuh(##32)

		memd(##32) = r15:14
# CHECK: 	memd(##32) = r15:14

		{r2 = #9; memw(##32) = r2.new}
# CHECK:	memw(##32) = r2.new

		{r2 = #9; memb(##32) = r2.new}
# CHECK:	memb(##32) = r2.new

		memw(##32) = r15
# CHECK: 	memw(##32) = r15

		memh(##32) = r16
# CHECK: 	memh(##32) = r16

		memb(##32) = r17
# CHECK: 	memb(##32) = r17


		r3:2 = interleave(r31:30)
# CHECK:	r3:2 = interleave(r31:30)
