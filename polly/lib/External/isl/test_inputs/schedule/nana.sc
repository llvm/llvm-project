# Check the decomposition of the topological sort.
# This decomposition is only performed by the incremental scheduler.
# OPTIONS: --no-schedule-whole-component
#
# This test case was contributed by Nana CK <1429802329@qq.com>.
domain: { sync[0:64, 0:32]; stB[0:64, 0:64]; stC[0:64, 0:64] }
validity: { stC[j = 0:64, 0:64] -> sync[j, 0]; stB[j = 0:64, 0:64] -> sync[j, 0] }
proximity: { stC[j = 0:64, 0:64] -> sync[j, 0]; stB[j = 0:64, 0:64] -> sync[j, 0] }
coincidence: { stC[j = 0:64, 0:64] -> sync[j, 0]; stB[j = 0:64, 0:64] -> sync[j, 0] }
