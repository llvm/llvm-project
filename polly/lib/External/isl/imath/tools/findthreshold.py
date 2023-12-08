#!/usr/bin/env python
##
## Name:     findthreshold.py
## Purpose:  Find a good threshold for recursive multiplication.
## Author:   M. J. Fromberger
##
## This tool computes some timing statistics to help you select a suitable
## recursive multiplication breakpoint.  It uses the imtimer tool to run a
## series of tests varying precision and breakpoint, and prints out a summary
## of the "best" values for each category.  Each summary line contains the
## following fields, tab-separated:
##
## prec     -- the precision of the operands (in digits).
## thresh   -- the threshold for recursive multiplication (digits).
## trec     -- total time using recursive algorithm (sec).
## tnorm    -- total time without recursive algorithm (sec).
## ratio    -- speedup (ratio of tnorm/trec).
##
## You are responsible for reading and interpreting the resulting table to
## obtain a useful value for your workload.  Change the default in imath.c, or
## call mp_int_multiply_threshold(n) during program initialization, to
## establish a satisfactory result.
##
import math, os, random, sys, time


def get_timing_stats(num_tests, precision, threshold, seed=None):
    """Obtain timing statistics for multiplication.

    num_tests      -- number of tests to run.
    precision      -- number of digits per operand.
    threshold      -- threshold in digits for recursive multiply.
    seed           -- random seed; if None, the clock is used.

    Returns a tuple of (seed, bits, time) where seed is the random seed used,
    bits is the number of bits per operand, and time is a float giving the
    total time taken for the test run.
    """
    if seed is None:
        seed = int(time.time())

    line = os.popen(
        './imtimer -mn -p %d -t %d -s %d %d' % (precision, threshold, seed,
                                                num_tests), 'r').readline()

    count, prec, bits, thresh, status = line.strip().split('\t')
    kind, total, unit = status.split()

    return seed, int(bits), float(total)


def check_binary(name):
    if not os.path.exists(name):
        os.system('make %s' % name)
        if not os.path.exists(name):
            raise ValueError("Unable to build %s" % name)
    elif not os.path.isfile(name):
        raise ValueError("Path exists with wrong type")


def compute_stats():
    check_binary('imtimer')
    seed = int(time.time())

    print >> sys.stderr, "Computing timer statistics (this may take a while)"
    stats = {}
    for prec in (32, 40, 64, 80, 128, 150, 256, 384, 512, 600, 768, 1024):
        sys.stderr.write('%-4d ' % prec)
        stats[prec] = (None, 1000000., 0.)

        for thresh in xrange(8, 65, 2):
            s, b, t = get_timing_stats(1000, prec, thresh, seed)
            sp, bp, tp = get_timing_stats(1000, prec, prec + 1, seed)

            if t < stats[prec][1]:
                stats[prec] = (thresh, t, tp)
                sys.stderr.write('+')
            else:
                sys.stderr.write('.')
        sys.stderr.write('\n')

    return list((p, h, t, tp) for p, (h, t, tp) in stats.iteritems())


if __name__ == "__main__":
    stats = compute_stats()
    stats.sort(key=lambda s: s[3] / s[2])
    for prec, thresh, trec, tnorm in stats:
        print "%d\t%d\t%.3f\t%.3f\t%.4f" % (prec, thresh, trec, tnorm,
                                            tnorm / trec)

    print

# Here there be dragons
