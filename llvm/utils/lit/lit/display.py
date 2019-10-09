import sys

import lit.ProgressBar

def create_display(opts, tests, total_tests, workers):
    if opts.quiet:
        return NopProgressDisplay()

    of_total = (' of %d' % total_tests) if (tests != total_tests) else ''
    header = '-- Testing: %d%s tests, %d workers --' % (tests, of_total, workers)

    progress_bar = None
    if opts.succinct and opts.useProgressBar:
        try:
            tc = lit.ProgressBar.TerminalController()
            progress_bar = lit.ProgressBar.ProgressBar(tc, header)
        except ValueError:
            print(header)
            progress_bar = lit.ProgressBar.SimpleProgressBar('Testing: ')
    else:
        print(header)

    if progress_bar:
        progress_bar.update(0, '')

    return ProgressDisplay(opts, tests, progress_bar)

class NopProgressDisplay(object):
    def update(self, test): pass
    def finish(self): pass

class ProgressDisplay(object):
    def __init__(self, opts, numTests, progressBar):
        self.opts = opts
        self.numTests = numTests
        self.progressBar = progressBar
        self.completed = 0

    def finish(self):
        if self.progressBar:
            self.progressBar.clear()
        elif self.opts.succinct:
            sys.stdout.write('\n')

    def update(self, test):
        self.completed += 1

        show_result = test.result.code.isFailure or \
                self.opts.showAllOutput or \
                (not self.opts.quiet and not self.opts.succinct)
        if show_result:
            self.print_result(test)

        if self.progressBar:
            percent = float(self.completed) / self.numTests
            self.progressBar.update(percent, test.getFullName())

    def print_result(self, test):
        if self.progressBar:
            self.progressBar.clear()

        # Show the test result line.
        test_name = test.getFullName()
        print('%s: %s (%d of %d)' % (test.result.code.name, test_name,
                                     self.completed, self.numTests))

        # Show the test failure output, if requested.
        if (test.result.code.isFailure and self.opts.showOutput) or \
           self.opts.showAllOutput:
            if test.result.code.isFailure:
                print("%s TEST '%s' FAILED %s" % ('*'*20, test.getFullName(),
                                                  '*'*20))
            print(test.result.output)
            print("*" * 20)

        # Report test metrics, if present.
        if test.result.metrics:
            print("%s TEST '%s' RESULTS %s" % ('*'*10, test.getFullName(),
                                               '*'*10))
            items = sorted(test.result.metrics.items())
            for metric_name, value in items:
                print('%s: %s ' % (metric_name, value.format()))
            print("*" * 10)

        # Report micro-tests, if present
        if test.result.microResults:
            items = sorted(test.result.microResults.items())
            for micro_test_name, micro_test in items:
                print("%s MICRO-TEST: %s" %
                         ('*'*3, micro_test_name))

                if micro_test.metrics:
                    sorted_metrics = sorted(micro_test.metrics.items())
                    for metric_name, value in sorted_metrics:
                        print('    %s:  %s ' % (metric_name, value.format()))

        # Ensure the output is flushed.
        sys.stdout.flush()
