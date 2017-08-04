#RUN: llvm-mc -triple=hexagon -filetype=obj %s | llvm-objdump -d - | FileCheck %s

.text
{
  r16 = memuh(r17 + #0)
  r18 = memuh(r19 + #0)
}

# CHECK: 289808ba
# CHECK: r16 = memuh(r17+#0);{{ *}}r18 = memuh(r19+#0)
