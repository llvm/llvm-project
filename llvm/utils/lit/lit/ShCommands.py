class Command:
    def __init__(self, args, redirects):
        self.args = list(args)
        self.redirects = list(redirects)

    def __repr__(self):
        return 'Command(%r, %r)' % (self.args, self.redirects)

    def __cmp__(self, other):
        if not isinstance(other, Command):
            return -1

        return cmp((self.args, self.redirects),
                   (other.args, other.redirects))

    def toShell(self, file):
        for arg in self.args:
            if "'" not in arg:
                quoted = "'%s'" % arg
            elif '"' not in arg and '$' not in arg:
                quoted = '"%s"' % arg
            else:
                raise NotImplementedError,'Unable to quote %r' % arg
            print >>file, quoted,

            # For debugging / validation.
            import ShUtil
            dequoted = list(ShUtil.ShLexer(quoted).lex())
            if dequoted != [arg]:
                raise NotImplementedError,'Unable to quote %r' % arg

        for r in self.redirects:
            if len(r[0]) == 1:
                print >>file, "%s '%s'" % (r[0][0], r[1]),
            else:
                print >>file, "%s%s '%s'" % (r[0][1], r[0][0], r[1]),

class Pipeline:
    def __init__(self, commands, negate=False, pipe_err=False):
        self.commands = commands
        self.negate = negate
        self.pipe_err = pipe_err

    def __repr__(self):
        return 'Pipeline(%r, %r, %r)' % (self.commands, self.negate,
                                         self.pipe_err)

    def __cmp__(self, other):
        if not isinstance(other, Pipeline):
            return -1

        return cmp((self.commands, self.negate, self.pipe_err),
                   (other.commands, other.negate, self.pipe_err))

    def toShell(self, file, pipefail=False):
        if pipefail != self.pipe_err:
            raise ValueError,'Inconsistent "pipefail" attribute!'
        if self.negate:
            print >>file, '!',
        for cmd in self.commands:
            cmd.toShell(file)
            if cmd is not self.commands[-1]:
                print >>file, '|\n ',

class Seq:
    def __init__(self, lhs, op, rhs):
        assert op in (';', '&', '||', '&&')
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self):
        return 'Seq(%r, %r, %r)' % (self.lhs, self.op, self.rhs)

    def __cmp__(self, other):
        if not isinstance(other, Seq):
            return -1

        return cmp((self.lhs, self.op, self.rhs),
                   (other.lhs, other.op, other.rhs))

    def toShell(self, file, pipefail=False):
        self.lhs.toShell(file, pipefail)
        print >>file, ' %s\n' % self.op
        self.rhs.toShell(file, pipefail)
