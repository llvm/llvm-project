## Test the cat command

# RUN: not %{lit} -a -v %{inputs}/shtest-cat \
# RUN: | FileCheck -match-full-lines %s
#
# END.

# CHECK: FAIL: shtest-cat :: cat-e.txt ({{[^)]*}})
# CHECK: cat -e {{.+}}/allchars | FileCheck {{.*}}
# CHECK: # executed command: cat -e {{.+}}/allchars
# CHECK: # | Unsupported: 'cat':  option -e not recognized
# CHECK: # error: command failed with exit status: {{.*}}
# CHECK: # executed command: FileCheck {{.*}}
# CHECK: # error: command failed with exit status: {{.*}}

# CHECK: PASS: shtest-cat :: cat-v.txt ({{[^)]*}})
# CHECK: cat -v {{.+}}/allchars | FileCheck {{.*}}
# CHECK-NEXT: # executed command: cat -v {{.+}}/allchars
# CHECK-NEXT: # executed command: FileCheck {{.*}}
# CHECK: cat -v {{.+}}/newline | FileCheck {{.*}}
# CHECK-NEXT: # executed command: cat -v {{.+}}/newline
# CHECK-NEXT: # executed command: FileCheck {{.*}}

# CHECK: PASS: shtest-cat :: cat.txt ({{[^)]*}})
# CHECK: cat {{.+}}/newline | FileCheck {{.*}}
# CHECK-NEXT: # executed command: cat {{.+}}/newline
# CHECK-NEXT: # executed command: FileCheck {{.*}}
# CHECK: cat {{.+}}/allchars > {{.+}}
# CHECK-NEXT: # executed command: cat {{.+}}/allchars
# CHECK: diff {{.+}}/allchars {{.+}}
# CHECK-NEXT: # executed command: diff {{.+}}/allchars {{.+}}