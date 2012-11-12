import itertools

from ShCommands import Command, Pipeline

def tcl_preprocess(data):
    # Tcl has a preprocessing step to replace escaped newlines.
    i = data.find('\\\n')
    if i == -1:
        return data

    # Replace '\\\n' and subsequent whitespace by a single space.
    n = len(data)
    str = data[:i]
    i += 2
    while i < n and data[i] in ' \t':
        i += 1
    return str + ' ' + data[i:]

class TclLexer:
    """TclLexer - Lex a string into "words", following the Tcl syntax."""

    def __init__(self, data):
        self.data = tcl_preprocess(data)
        self.pos = 0
        self.end = len(self.data)

    def at_end(self):
        return self.pos == self.end

    def eat(self):
        c = self.data[self.pos]
        self.pos += 1
        return c

    def look(self):
        return self.data[self.pos]

    def maybe_eat(self, c):
        """
        maybe_eat(c) - Consume the character c if it is the next character,
        returning True if a character was consumed. """
        if self.data[self.pos] == c:
            self.pos += 1
            return True
        return False

    def escape(self, c):
        if c == 'a':
            return '\x07'
        elif c == 'b':
            return '\x08'
        elif c == 'f':
            return '\x0c'
        elif c == 'n':
            return '\n'
        elif c == 'r':
            return '\r'
        elif c == 't':
            return '\t'
        elif c == 'v':
            return '\x0b'
        elif c in 'uxo':
            raise ValueError,'Invalid quoted character %r' % c
        else:
            return c
        
    def lex_braced(self):
        # Lex until whitespace or end of string, the opening brace has already
        # been consumed.

        str = ''        
        while 1:
            if self.at_end():
                raise ValueError,"Unterminated '{' quoted word"
            
            c = self.eat()
            if c == '}':
                break
            elif c == '{':
                str += '{' + self.lex_braced() + '}'
            elif c == '\\' and self.look() in '{}':
                str += self.eat()
            else:
                str += c

        return str

    def lex_quoted(self):
        str = ''

        while 1:
            if self.at_end():
                raise ValueError,"Unterminated '\"' quoted word"
            
            c = self.eat()
            if c == '"':
                break
            elif c == '\\':
                if self.at_end():
                    raise ValueError,'Missing quoted character'

                str += self.escape(self.eat())
            else:
                str += c

        return str

    def lex_unquoted(self, process_all=False):
        # Lex until whitespace or end of string.
        str = ''
        while not self.at_end():
            if not process_all:
                if self.look().isspace() or self.look() == ';':
                    break

            c = self.eat()
            if c == '\\':
                if self.at_end():
                    raise ValueError,'Missing quoted character'

                str += self.escape(self.eat())
            elif c == '[':
                raise NotImplementedError, ('Command substitution is '
                                            'not supported')
            elif c == '$' and not self.at_end() and (self.look().isalpha() or
                                                     self.look() == '{'):
                raise NotImplementedError, ('Variable substitution is '
                                            'not supported')
            else:
                str += c

        return str

    def lex_one_token(self):
        if self.maybe_eat('"'):
            return self.lex_quoted()
        elif self.maybe_eat('{'):
            # Check for argument substitution.
            if not self.maybe_eat('*'):
                return self.lex_braced()

            if not self.maybe_eat('}'):
                    return '*' + self.lex_braced()
                
            if self.at_end() or self.look().isspace():
                return '*'

            raise NotImplementedError, "Argument substitution is unsupported"
        else:
            return self.lex_unquoted()

    def lex(self):
        while not self.at_end():
            c = self.look()
            if c in ' \t':
                self.eat()
            elif c in ';\n':
                self.eat()
                yield (';',)
            else:
                yield self.lex_one_token()

class TclExecCommand:
    kRedirectPrefixes1 = ('<', '>')
    kRedirectPrefixes2 = ('<@', '<<', '2>', '>&', '>>', '>@')
    kRedirectPrefixes3 = ('2>@', '2>>', '>>&', '>&@')
    kRedirectPrefixes4 = ('2>@1',)

    def __init__(self, args):
        self.args = iter(args)

    def lex(self):
        try:
            return self.args.next()
        except StopIteration:
            return None

    def look(self):
        next = self.lex()
        if next is not None:
            self.args = itertools.chain([next], self.args)
        return next

    def parse_redirect(self, tok, length):
        if len(tok) == length:
            arg = self.lex()
            if arg is None:
                raise ValueError,'Missing argument to %r redirection' % tok
        else:
            tok,arg = tok[:length],tok[length:]

        if tok[0] == '2':
            op = (tok[1:],2)
        else:
            op = (tok,)
        return (op, arg)

    def parse_pipeline(self):
        if self.look() is None:
            raise ValueError,"Expected at least one argument to exec"

        commands = [Command([],[])]
        while 1:
            arg = self.lex()
            if arg is None:
                break
            elif arg == '|':
                commands.append(Command([],[]))
            elif arg == '|&':
                # Write this as a redirect of stderr; it must come first because
                # stdout may have already been redirected.
                commands[-1].redirects.insert(0, (('>&',2),'1'))
                commands.append(Command([],[]))
            elif arg[:4] in TclExecCommand.kRedirectPrefixes4:
                commands[-1].redirects.append(self.parse_redirect(arg, 4))
            elif arg[:3] in TclExecCommand.kRedirectPrefixes3:
                commands[-1].redirects.append(self.parse_redirect(arg, 3))
            elif arg[:2] in TclExecCommand.kRedirectPrefixes2:
                commands[-1].redirects.append(self.parse_redirect(arg, 2))
            elif arg[:1] in TclExecCommand.kRedirectPrefixes1:
                commands[-1].redirects.append(self.parse_redirect(arg, 1))
            else:
                commands[-1].args.append(arg)

        return Pipeline(commands, False, pipe_err=True)

    def parse(self):
        ignoreStderr = False
        keepNewline = False

        # Parse arguments.
        while 1:
            next = self.look()
            if not isinstance(next, str) or next[0] != '-':
                break

            if next == '--':
                self.lex()
                break
            elif next == '-ignorestderr':
                ignoreStderr = True
            elif next == '-keepnewline':
                keepNewline = True
            else:
                raise ValueError,"Invalid exec argument %r" % next

        return (ignoreStderr, keepNewline, self.parse_pipeline())

###

import unittest

class TestTclLexer(unittest.TestCase):
    def lex(self, str, *args, **kwargs):
        return list(TclLexer(str, *args, **kwargs).lex())

    def test_preprocess(self):
        self.assertEqual(tcl_preprocess('a b'), 'a b')
        self.assertEqual(tcl_preprocess('a\\\nb c'), 'a b c')

    def test_unquoted(self):
        self.assertEqual(self.lex('a b c'),
                         ['a', 'b', 'c'])
        self.assertEqual(self.lex(r'a\nb\tc\ '),
                         ['a\nb\tc '])
        self.assertEqual(self.lex(r'a \\\$b c $\\'),
                         ['a', r'\$b', 'c', '$\\'])

    def test_braced(self):
        self.assertEqual(self.lex('a {b c} {}'),
                         ['a', 'b c', ''])
        self.assertEqual(self.lex(r'a {b {c\n}}'),
                         ['a', 'b {c\\n}'])
        self.assertEqual(self.lex(r'a {b\{}'),
                         ['a', 'b{'])
        self.assertEqual(self.lex(r'{*}'), ['*'])
        self.assertEqual(self.lex(r'{*} a'), ['*', 'a'])
        self.assertEqual(self.lex(r'{*} a'), ['*', 'a'])
        self.assertEqual(self.lex('{a\\\n   b}'),
                         ['a b'])

    def test_quoted(self):
        self.assertEqual(self.lex('a "b c"'),
                         ['a', 'b c'])

    def test_terminators(self):
        self.assertEqual(self.lex('a\nb'),
                         ['a', (';',), 'b'])
        self.assertEqual(self.lex('a;b'),
                         ['a', (';',), 'b'])
        self.assertEqual(self.lex('a   ;   b'),
                         ['a', (';',), 'b'])

class TestTclExecCommand(unittest.TestCase):
    def parse(self, str):
        return TclExecCommand(list(TclLexer(str).lex())).parse()

    def test_basic(self):
        self.assertEqual(self.parse('echo hello'),
                         (False, False,
                          Pipeline([Command(['echo', 'hello'], [])],
                                   False, True)))
        self.assertEqual(self.parse('echo hello | grep hello'),
                         (False, False,
                          Pipeline([Command(['echo', 'hello'], []),
                                    Command(['grep', 'hello'], [])],
                                   False, True)))

    def test_redirect(self):
        self.assertEqual(self.parse('echo hello > a >b >>c 2> d |& e'),
                         (False, False,
                          Pipeline([Command(['echo', 'hello'],
                                            [(('>&',2),'1'),
                                             (('>',),'a'),
                                             (('>',),'b'),
                                             (('>>',),'c'),
                                             (('>',2),'d')]),
                                    Command(['e'], [])],
                                   False, True)))

if __name__ == '__main__':
    unittest.main()
