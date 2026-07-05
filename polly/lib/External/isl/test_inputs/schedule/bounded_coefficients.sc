# Check that the bounds on the coefficients are respected.
# This function checks for a particular output schedule,
# but the exact output is not important, only that it does
# not contain any coefficients greater than 4.
# It is, however, easier to check for a particular output.
# This test uses the whole component scheduler
# because the incremental scheduler has no reason to fuse anything.
# OPTIONS: --schedule-whole-component  --schedule-max-coefficient=4 --schedule-max-constant-term=10
domain: { S_4[i, j, k] : 0 <= i < j <= 10 and 0 <= k <= 100;
	  S_2[i, j] : 0 <= i < j <= 10; S_6[i, j] : 0 <= i < j <= 10 }
validity: { S_2[0, j] -> S_4[0, j, 0] : 0 < j <= 10;
	    S_4[0, j, 100] -> S_6[0, j] : 0 < j <= 10 }
