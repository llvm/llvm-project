# Check the cat command

# RUN: not %{lit} -a -v %{inputs}/shtest-cat \
# RUN: | FileCheck -match-full-lines %s
#
# END.

# CHECK: FAIL: shtest-cat :: cat-e-allchars.txt ({{[^)]*}})
# CHECK: cat -e {{.+}}/allchars | FileCheck {{.*}}
# CHECK: # executed command: cat -e {{.*}}
# CHECK: # | Unsupported: 'cat':  option -e not recognized
# CHECK: # error: command failed with exit status: {{.*}}
# CHECK: # executed command: FileCheck {{.*}}
# CHECK: # error: command failed with exit status: {{.*}}

# CHECK: FAIL: shtest-cat :: cat-e-newline.txt ({{[^)]*}})
# CHECK: cat -e {{.+}}/newline | FileCheck {{.*}}
# CHECK: # executed command: cat -e {{.*}}
# CHECK: # | Unsupported: 'cat':  option -e not recognized
# CHECK: # error: command failed with exit status: {{.*}}
# CHECK: # executed command: FileCheck {{.*}}
# CHECK: # error: command failed with exit status: {{.*}}

# CHECK: PASS: shtest-cat :: cat-newline.txt ({{[^)]*}})
# CHECK: cat {{.+}}/newline | FileCheck {{.*}}
# CHECK: # executed command: cat {{.*}}
# CHECK: # executed command: FileCheck {{.*}}

# CHECK: PASS: shtest-cat :: cat-v-allchars.txt ({{[^)]*}})
# CHECK: cat -v {{.+}}/allchars | FileCheck {{.*}}
# CHECK: # executed command: cat -v {{.*}}
# CHECK: # executed command: FileCheck {{.*}}