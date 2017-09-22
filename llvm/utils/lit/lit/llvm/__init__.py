from lit.llvm import config
import lit.util
import re

llvm_config = None

class ToolFilter(object):
    """
        String-like class used to build regex substitution patterns for
        llvm tools.  Handles things like adding word-boundary patterns,
        and filtering characters from the beginning an end of a tool name
    """

    def __init__(self, name, pre=None, post=None, verbatim=False):
        """
            Construct a ToolFilter.

            name: the literal name of the substitution to look for.

            pre: If specified, the substitution will not find matches where
            the character immediately preceding the word-boundary that begins
            `name` is any of the characters in the string `pre`.

            post: If specified, the substitution will not find matches where
            the character immediately after the word-boundary that ends `name`
            is any of the characters specified in the string `post`.

            verbatim: If True, `name` is an exact regex that is passed to the
            underlying substitution
        """
        if verbatim:
            self.regex = name
            return

        def not_in(chars, where=''):
            if not chars:
                return ''
            pattern_str = '|'.join(re.escape(x) for x in chars)
            return r'(?{}!({}))'.format(where, pattern_str)

        self.regex = not_in(pre, '<') + r'\b' + name + r'\b' + not_in(post)

    def __str__(self):
        return self.regex


def initialize(lit_config, test_config):
    global llvm_config

    llvm_config = config.LLVMConfig(lit_config, test_config)

