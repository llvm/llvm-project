
# Classes responsible for drawing signs in the Vim user interface.

import vim


class VimSign(object):
    SIGN_TEXT_BREAKPOINT_RESOLVED = "B>"
    SIGN_TEXT_BREAKPOINT_UNRESOLVED = "b>"
    SIGN_TEXT_PC = "->"
    SIGN_HIGHLIGHT_COLOUR_PC = 'darkblue'

    # unique sign id (for ':[sign/highlight] define)
    sign_id = 1

    # unique name id (for ':sign place')
    name_id = 1

    # Map of {(sign_text, highlight_colour) --> sign_name}
    defined_signs = {}

    def __init__(self, sign_text, buffer, line_number, highlight_colour=None):
        """ Define the sign and highlight (if applicable) and show the sign. """

        # Get the sign name, either by defining it, or looking it up in the map
        # of defined signs
        key = (sign_text, highlight_colour)
        if key not in VimSign.defined_signs:
            name = self.define(sign_text, highlight_colour)
        else:
            name = VimSign.defined_signs[key]

        self.show(name, buffer.number, line_number)
        pass

    def define(self, sign_text, highlight_colour):
        """ Defines sign and highlight (if highlight_colour is not None). """
        sign_name = "sign%d" % VimSign.name_id
        if highlight_colour is None:
            vim.command("sign define %s text=%s" % (sign_name, sign_text))
        else:
            self.highlight_name = "highlight%d" % VimSign.name_id
            vim.command(
                "highlight %s ctermbg=%s guibg=%s" %
                (self.highlight_name, highlight_colour, highlight_colour))
            vim.command(
                "sign define %s text=%s linehl=%s texthl=%s" %
                (sign_name, sign_text, self.highlight_name, self.highlight_name))
        VimSign.defined_signs[(sign_text, highlight_colour)] = sign_name
        VimSign.name_id += 1
        return sign_name

    def show(self, name, buffer_number, line_number):
        self.id = VimSign.sign_id
        VimSign.sign_id += 1
        vim.command("sign place %d name=%s line=%d buffer=%s" %
                    (self.id, name, line_number, buffer_number))
        pass

    def hide(self):
        vim.command("sign unplace %d" % self.id)
        pass


class BreakpointSign(VimSign):

    def __init__(self, buffer, line_number, is_resolved):
        txt = VimSign.SIGN_TEXT_BREAKPOINT_RESOLVED if is_resolved else VimSign.SIGN_TEXT_BREAKPOINT_UNRESOLVED
        super(BreakpointSign, self).__init__(txt, buffer, line_number)


class PCSign(VimSign):

    def __init__(self, buffer, line_number, is_selected_thread):
        super(
            PCSign,
            self).__init__(
            VimSign.SIGN_TEXT_PC,
            buffer,
            line_number,
            VimSign.SIGN_HIGHLIGHT_COLOUR_PC if is_selected_thread else None)
