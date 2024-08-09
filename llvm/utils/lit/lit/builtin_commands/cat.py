import getopt
import sys
from dataclasses import dataclass

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


# This dataclass defines all currently supported options for cat
@dataclass
class Options:
    # Options: -e. True if newlines should be displayed with a '$'
    show_ends: bool
    # Options: -v, -e. True if text should be converted to ^ and M- notation
    show_nonprinting: bool


def convertTextNotation(data, options):
    assert options.show_ends or options.show_nonprinting

    newdata = StringIO()
    if isinstance(data, str):
        data = bytearray(data.encode())

    for intval in data:
        if intval == 10 and options.show_ends:
            newdata.write("$")
            newdata.write(chr(intval))
            continue
        if options.show_nonprinting:
            if intval == 9 or intval == 10:
                newdata.write(chr(intval))
                continue
            if intval > 127:
                intval = intval - 128
                newdata.write("M-")
            if intval < 32:
                newdata.write("^")
                newdata.write(chr(intval + 64))
                continue
            elif intval == 127:
                newdata.write("^?")
                continue
        newdata.write(chr(intval))

    return newdata.getvalue().encode()


def main(argv):
    arguments = argv[1:]
    short_options = "eEv"
    long_options = ["show-ends", "show-nonprinting"]
    enabled_options = Options(show_ends=False, show_nonprinting=False)
    convert_text = False

    try:
        options, filenames = getopt.gnu_getopt(arguments, short_options, long_options)
    except getopt.GetoptError as err:
        sys.stderr.write("Unsupported: 'cat':  %s\n" % str(err))
        sys.exit(1)

    for option, value in options:
        if option == "-v" or option == "--show-nonprinting" or option == "-e":
            enabled_options.show_nonprinting = True
            convert_text = True
        if option == "-E" or option == "--show-ends" or option == "-e":
            enabled_options.show_ends = True
            convert_text = True

    writer = getattr(sys.stdout, "buffer", None)
    if writer is None:
        writer = sys.stdout
        if sys.platform == "win32":
            import os, msvcrt

            msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    for filename in filenames:
        try:
            contents = None
            is_text = False
            try:
                if sys.platform != "win32":
                    fileToCat = open(filename, "r")
                    contents = fileToCat.read()
                    is_text = True
            except:
                pass

            if contents is None:
                fileToCat = open(filename, "rb")
                contents = fileToCat.read()

            if convert_text:
                contents = convertTextNotation(contents, enabled_options)
            elif is_text:
                contents = contents.encode()
            writer.write(contents)
            sys.stdout.flush()
            fileToCat.close()
        except IOError as error:
            sys.stderr.write(str(error))
            sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)
