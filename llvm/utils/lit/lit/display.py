import sys


def create_display(opts, tests, total_tests, workers):
    if opts.quiet:
        return NopDisplay()

    of_total = (' of %d' % total_tests) if (tests != total_tests) else ''
    header = '-- Testing: %d%s tests, %d workers --' % (tests, of_total, workers)

    progress_bar = None
    if opts.succinct and opts.useProgressBar:
        import lit.ProgressBar
        try:
            tc = lit.ProgressBar.TerminalController()
            progress_bar = lit.ProgressBar.ProgressBar(tc, header)
            header = None
        except ValueError:
            progress_bar = lit.ProgressBar.SimpleProgressBar('Testing: ')

    return Display(opts, tests, header, progress_bar)


class NopDisplay(object):
    def print_header(self): pass
    def update(self, test): pass
    def clear(self, interrupted): pass


class Display(object):
    def __init__(self, opts, tests, header, progress_bar):
        self.opts = opts
        self.tests = tests
        self.header = header
        self.progress_bar = progress_bar
        self.completed = 0

    def print_header(self):
        if self.header:
            print(self.header)
        if self.progress_bar:
            self.progress_bar.update(0.0, '')

    def update(self, test):
        self.completed += 1

        show_result = test.isFailure() or \
                self.opts.showAllOutput or \
                (not self.opts.quiet and not self.opts.succinct)
        if show_result:
            if self.progress_bar:
                self.progress_bar.clear(interrupted=False)
            self.print_result(test)

        if self.progress_bar:
            if test.isFailure():
                self.progress_bar.barColor = 'RED'
            percent = float(self.completed) / self.tests
            self.progress_bar.update(percent, test.getFullName())

    def clear(self, interrupted):
        if self.progress_bar:
            self.progress_bar.clear(interrupted)

    def print_result(self, test):
        # Show the test result line.
        test_name = test.getFullName()
        print('%s: %s (%d of %d)' % (test.result.code.name, test_name,
                                     self.completed, self.tests))

        # Show the test failure output, if requested.
        if (test.isFailure() and self.opts.showOutput) or \
           self.opts.showAllOutput:
            if test.isFailure():
                print("%s TEST '%s' FAILED %s" % ('*'*20, test.getFullName(),
                                                  '*'*20))
            out = test.result.output
            # Encode/decode so that, when using Python 3.6.5 in Windows 10,
            # print(out) doesn't raise UnicodeEncodeError if out contains
            # special characters.  However, Python 2 might try to decode
            # as part of the encode call if out is already encoded, so skip
            # encoding if it raises UnicodeDecodeError.
            if sys.stdout.encoding:
                try:
                    out = out.encode(encoding=sys.stdout.encoding,
                                     errors="replace")
                except UnicodeDecodeError:
                    pass
                out = out.decode(encoding=sys.stdout.encoding)
            print(out)
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
