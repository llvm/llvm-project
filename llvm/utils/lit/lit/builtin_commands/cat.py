import getopt
import sys
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

def convertToCaretAndMNotation(data):
   newdata = StringIO()
   for char in data:
       intval = ord(char)
       if intval == 9 or intval == 10:
           newdata.write(chr(intval))
           continue
       if intval > 127:
           intval = intval -128
           newdata.write("M-")
       if intval < 32:
           newdata.write("^")
           newdata.write(chr(intval+64))
       elif intval == 127:
           newdata.write("^?")
       else:
           newdata.write(chr(intval))

   return newdata.getvalue();


def main(argv):
    arguments = argv[1:]
    short_options = "v"
    long_options = ["show-nonprinting"]
    show_nonprinting = False;

    try:
        options, filenames = getopt.gnu_getopt(arguments, short_options, long_options)
    except getopt.GetoptError as err:
        sys.stderr.write("Unsupported: 'cat':  %s\n" % str(err))
        sys.exit(1)

    for option, value in options:
        if option == "-v" or option == "--show-nonprinting":
            show_nonprinting = True;

    for filename in filenames:
        try:
            fileToCat = open(filename,"rb")
            contents = fileToCat.read()
            if show_nonprinting:
                contents = convertToCaretAndMNotation(contents)
            sys.stdout.write(contents)
            sys.stdout.flush()
            fileToCat.close()
        except IOError as error:
            sys.stderr.write(str(error))
            sys.exit(1)

if __name__ == "__main__":
    main(sys.argv)
