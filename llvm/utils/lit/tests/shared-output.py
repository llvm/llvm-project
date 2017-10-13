# RUN: rm -rf %t && mkdir -p %t
# RUN: echo 'lit_config.load_config(config, "%{inputs}/shared-output/lit.cfg")' > %t/lit.site.cfg
# RUN: %{lit} %t
# RUN: FileCheck %s < %t/Output/Shared/SHARED.tmp
# RUN: FileCheck -check-prefix=NEGATIVE %s < %t/Output/Shared/SHARED.tmp
# RUN: FileCheck -check-prefix=OTHER %s <  %t/Output/Shared/OTHER.tmp

# CHECK-DAG: primary
# CHECK-DAG: secondary
# CHECK-DAG: sub

# NEGATIVE-NOT: other
# OTHER: other
