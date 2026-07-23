import getopt
import os
import platform
import sys
from io import StringIO


def convertToCaretAndMNotation(data):
    newdata = StringIO()
    if isinstance(data, str):
        data = bytearray(data.encode())

    for intval in data:
        if intval == 9 or intval == 10:
            newdata.write(chr(intval))
            continue
        if intval > 127:
            intval = intval - 128
            newdata.write("M-")
        if intval < 32:
            newdata.write("^")
            newdata.write(chr(intval + 64))
        elif intval == 127:
            newdata.write("^?")
        else:
            newdata.write(chr(intval))

    return newdata.getvalue().encode()


def run(argv, stdin, stdout, stderr, cwd):
    """In-process cat.

    Since it never calls sys.exit, this is safe to run inside the lit worker as well as
    from the standalone main below.

    Args:
        argv: A list of command-line arguments. The first element is the command name,
            followed by options or filenames.
        stdin: Binary input stream used if no files are specified.
        stdout: Binary output stream for the concatenated content.
        stderr: Binary error stream for error messages.
        cwd: The shell's current working directory, used to resolve relative file paths.

    Returns:
        An integer representing the exit code (0 for success, 1 for errors).
    """
    arguments = argv[1:]
    short_options = "v"
    long_options = ["show-nonprinting"]
    show_nonprinting = False

    try:
        options, filenames = getopt.gnu_getopt(arguments, short_options, long_options)
    except getopt.GetoptError as err:
        stderr.write(b"Unsupported: 'cat':  %s\n" % str(err).encode())
        return 1

    for option, value in options:
        if option == "-v" or option == "--show-nonprinting":
            show_nonprinting = True

    if len(filenames) == 0:
        stdout.write(stdin.read())
        return 0

    for filename in filenames:
        path = filename
        if not os.path.isabs(path):
            path = os.path.join(cwd, path)

        contents = None
        is_text = False
        if platform.system() == "OS/390":
            try:
                with open(path, "r") as fileToCat:
                    contents = fileToCat.read()
                    is_text = True
            except:
                pass

        if contents is None:
            try:
                with open(path, "rb") as fileToCat:
                    contents = fileToCat.read()
            except IOError as error:
                error.filename = filename
                stderr.write(str(error).encode())
                return 1

        if show_nonprinting:
            contents = convertToCaretAndMNotation(contents)
        elif is_text:
            contents = contents.encode()
        stdout.write(contents)
    return 0


def main(argv):
    out = getattr(sys.stdout, "buffer", sys.stdout)
    sys.exit(run(argv, sys.stdin.buffer, out, sys.stderr.buffer, os.getcwd()))


if __name__ == "__main__":
    main(sys.argv)
