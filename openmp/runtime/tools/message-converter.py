#!/usr/bin/env python3

#
# //===----------------------------------------------------------------------===//
# //
# // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# // See https://llvm.org/LICENSE.txt for license information.
# // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# //
# //===----------------------------------------------------------------------===//
#

import argparse
import datetime
import os
import platform
import re
import sys
from libomputils import ScriptError, error


class TargetPlatform:
    """Convenience class for handling the target platform for configuration/compilation"""

    system_override = None
    """
    Target system name override by the user.
    It follows the conventions from https://docs.python.org/3/library/platform.html#platform.system
    """

    def set_system_override(override_system):
        """
        Set a system override for the target.
        Please follow the style from https://docs.python.org/3/library/platform.html#platform.system
        """
        TargetPlatform.system_override = override_system

    def system():
        """
        Target System name.
        It follows the conventions from https://docs.python.org/3/library/platform.html#platform.system
        """
        if TargetPlatform.system_override is None:
            return platform.system()
        return TargetPlatform.system_override


class ParseMessageDataError(ScriptError):
    """Convenience class for parsing message data file errors"""

    def __init__(self, filename, line, msg):
        super(ParseMessageDataError, self).__init__(msg)
        self.filename = filename
        self.line = line


def parse_error(filename, line, msg):
    raise ParseMessageDataError(filename, line, msg)


def display_language_id(inputFile):
    """Quickly parse file for LangId and print it"""
    regex = re.compile(r'^LangId\s+"([0-9]+)"')
    with open(inputFile, encoding="utf-8") as f:
        for line in f:
            m = regex.search(line)
            if not m:
                continue
            print(m.group(1))


class Message(object):
    special = {
        "n": "\n",
        "t": "\t",
    }

    def __init__(self, lineNumber, name, text):
        self.lineNumber = lineNumber
        self.name = name
        self.text = text

    def toSrc(self):
        if TargetPlatform.system().casefold() == "Windows".casefold():
            return re.sub(r"%([0-9])\$(s|l?[du])", r"%\1!\2!", self.text)
        return str(self.text)

    def toMC(self):
        retval = self.toSrc()
        for special, substitute in Message.special.items():
            retval = re.sub(r"\\{}".format(special), substitute, retval)
        return retval


class MessageData(object):
    """
    Convenience class representing message data parsed from i18n/* files

    Generate these objects using static create() factory method
    """

    sectionInfo = {
        "meta": {"short": "prp", "long": "meta", "set": 1, "base": 1 << 16},
        "strings": {"short": "str", "long": "strings", "set": 2, "base": 2 << 16},
        "formats": {"short": "fmt", "long": "formats", "set": 3, "base": 3 << 16},
        "messages": {"short": "msg", "long": "messages", "set": 4, "base": 4 << 16},
        "hints": {"short": "hnt", "long": "hints", "set": 5, "base": 5 << 16},
    }
    orderedSections = ["meta", "strings", "formats", "messages", "hints"]

    def __init__(self):
        self.filename = None
        self.sections = {}

    def getMeta(self, name):
        metaList = self.sections["meta"]
        for meta in metaList:
            if meta.name == name:
                return meta.text
        error(
            'No "{}" detected in meta data' " for file {}".format(name, self.filename)
        )

    @staticmethod
    def create(inputFile):
        """Creates MessageData object from inputFile"""
        data = MessageData()
        data.filename = os.path.abspath(inputFile)
        obsolete = 1
        sectionRegex = re.compile(r"-\*- ([a-zA-Z0-9_]+) -\*-")
        keyValueRegex = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)\s+"(.*)"')
        moreValueRegex = re.compile(r'"(.*)"')

        with open(inputFile, "r", encoding="utf-8") as f:
            currentSection = None
            currentKey = None
            for lineNumber, line in enumerate(f, 1):
                line = line.strip()
                # Skip empty lines
                if not line:
                    continue
                # Skip comment lines
                if line.startswith("#"):
                    continue
                # Matched a section header
                match = sectionRegex.search(line)
                if match:
                    currentSection = match.group(1).lower()
                    if currentSection in data.sections:
                        parse_error(
                            inputFile,
                            lineNumber,
                            "section: {} already defined".format(currentSection),
                        )
                    data.sections[currentSection] = []
                    continue
                # Matched a Key "Value" line (most lines)
                match = keyValueRegex.search(line)
                if match:
                    if not currentSection:
                        parse_error(inputFile, lineNumber, "no section defined yet.")
                    key = match.group(1)
                    if key == "OBSOLETE":
                        key = "OBSOLETE{}".format(obsolete)
                        obsolete += 1
                    value = match.group(2)
                    currentKey = key
                    data.sections[currentSection].append(
                        Message(lineNumber, key, value)
                    )
                    continue
                # Matched a Continuation of string line
                match = moreValueRegex.search(line)
                if match:
                    value = match.group(1)
                    if not currentSection:
                        parse_error(inputFile, lineNumber, "no section defined yet.")
                    if not currentKey:
                        parse_error(inputFile, lineNumber, "no key defined yet.")
                    data.sections[currentSection][-1].text += value
                    continue
                # Unknown line syntax
                parse_error(inputFile, lineNumber, "bad line:\n{}".format(line))
        return data


def insert_header(f, data, commentChar="//"):
    f.write(
        "{0} Do not edit this file! {0}\n"
        "{0} The file was generated from"
        " {1} by {2} on {3}. {0}\n\n".format(
            commentChar,
            os.path.basename(data.filename),
            os.path.basename(__file__),
            datetime.datetime.now().ctime(),
        )
    )


def generate_enum_file(enumFile, prefix, data):
    """Create the include file with message enums"""
    global g_sections
    with open(enumFile, "w") as f:
        insert_header(f, data)
        f.write(
            "enum {0}_id {1}\n"
            "\n"
            "    // A special id for absence of message.\n"
            "    {0}_null = 0,\n"
            "\n".format(prefix, "{")
        )
        for section in MessageData.orderedSections:
            messages = data.sections[section]
            info = MessageData.sectionInfo[section]
            shortName = info["short"]
            longName = info["long"]
            base = info["base"]
            setIdx = info["set"]
            f.write(
                "    // Set #{}, {}.\n"
                "    {}_{}_first = {},\n".format(
                    setIdx, longName, prefix, shortName, base
                )
            )
            for message in messages:
                f.write("    {}_{}_{},\n".format(prefix, shortName, message.name))
            f.write("    {}_{}_last,\n\n".format(prefix, shortName))
        f.write(
            "    {0}_xxx_lastest\n\n"
            "{1}; // enum {0}_id\n\n"
            "typedef enum {0}_id  {0}_id_t;\n\n\n"
            "// end of file //\n".format(prefix, "}")
        )


def generate_signature_file(signatureFile, data):
    """Create the signature file"""
    sigRegex = re.compile(r"(%[0-9]\$(s|l?[du]))")
    with open(signatureFile, "w") as f:
        f.write("// message catalog signature file //\n\n")
        for section in MessageData.orderedSections:
            messages = data.sections[section]
            longName = MessageData.sectionInfo[section]["long"]
            f.write("-*- {}-*-\n\n".format(longName.upper()))
            for message in messages:
                sigs = sorted(list(set([a for a, b in sigRegex.findall(message.text)])))
                i = 0
                # Insert empty placeholders if necessary
                while i != len(sigs):
                    num = i + 1
                    if not sigs[i].startswith("%{}".format(num)):
                        sigs.insert(i, "%{}$-".format(num))
                    else:
                        i += 1
                f.write("{:<40} {}\n".format(message.name, " ".join(sigs)))
            f.write("\n")
        f.write("// end of file //\n")


def generate_default_messages_file(defaultFile, prefix, data):
    """Create the include file with message strings organized"""
    with open(defaultFile, "w", encoding="utf-8") as f:
        insert_header(f, data)
        for section in MessageData.orderedSections:
            f.write(
                "static char const *\n"
                "__{}_default_{}[] =\n"
                "    {}\n"
                "        NULL,\n".format(prefix, section, "{")
            )
            messages = data.sections[section]
            for message in messages:
                f.write('        "{}",\n'.format(message.toSrc()))
            f.write("        NULL\n" "    {};\n\n".format("}"))
        f.write(
            "struct kmp_i18n_section {0}\n"
            "    int           size;\n"
            "    char const ** str;\n"
            "{1}; // struct kmp_i18n_section\n"
            "typedef struct kmp_i18n_section  kmp_i18n_section_t;\n\n"
            "static kmp_i18n_section_t\n"
            "__{2}_sections[] =\n"
            "    {0}\n"
            "        {0} 0, NULL {1},\n".format("{", "}", prefix)
        )

        for section in MessageData.orderedSections:
            messages = data.sections[section]
            f.write(
                "        {} {}, __{}_default_{} {},\n".format(
                    "{", len(messages), prefix, section, "}"
                )
            )
        numSections = len(MessageData.orderedSections)
        f.write(
            "        {0} 0, NULL {1}\n"
            "    {1};\n\n"
            "struct kmp_i18n_table {0}\n"
            "    int                   size;\n"
            "    kmp_i18n_section_t *  sect;\n"
            "{1}; // struct kmp_i18n_table\n"
            "typedef struct kmp_i18n_table  kmp_i18n_table_t;\n\n"
            "static kmp_i18n_table_t __kmp_i18n_default_table =\n"
            "    {0}\n"
            "        {3},\n"
            "        __{2}_sections\n"
            "    {1};\n\n"
            "// end of file //\n".format("{", "}", prefix, numSections)
        )


def generate_message_file_unix(messageFile, data):
    """
    Create the message file for Unix OSes

    Encoding is in UTF-8
    """
    with open(messageFile, "w", encoding="utf-8") as f:
        insert_header(f, data, commentChar="$")
        f.write('$quote "\n\n')
        for section in MessageData.orderedSections:
            setIdx = MessageData.sectionInfo[section]["set"]
            f.write(
                "$ ------------------------------------------------------------------------------\n"
                "$ {}\n"
                "$ ------------------------------------------------------------------------------\n\n"
                "$set {}\n\n".format(section, setIdx)
            )
            messages = data.sections[section]
            for num, message in enumerate(messages, 1):
                f.write('{} "{}"\n'.format(num, message.toSrc()))
            f.write("\n")
        f.write("\n$ end of file $")


def generate_message_file_windows(messageFile, data):
    """
    Create the message file for Windows OS

    Encoding is in UTF-16LE
    """
    language = data.getMeta("Language")
    langId = data.getMeta("LangId")
    with open(messageFile, "w", encoding="utf-16-le") as f:
        insert_header(f, data, commentChar=";")
        f.write("\nLanguageNames = ({0}={1}:msg_{1})\n\n".format(language, langId))
        f.write("FacilityNames=(\n")
        for section in MessageData.orderedSections:
            setIdx = MessageData.sectionInfo[section]["set"]
            shortName = MessageData.sectionInfo[section]["short"]
            f.write(" {}={}\n".format(shortName, setIdx))
        f.write(")\n\n")

        for section in MessageData.orderedSections:
            shortName = MessageData.sectionInfo[section]["short"]
            n = 0
            messages = data.sections[section]
            for message in messages:
                n += 1
                f.write(
                    "MessageId={}\n"
                    "Facility={}\n"
                    "Language={}\n"
                    "{}\n.\n\n".format(n, shortName, language, message.toMC())
                )
        f.write("\n; end of file ;")


def main():
    parser = argparse.ArgumentParser(description="Generate message data files")
    parser.add_argument(
        "--lang-id",
        action="store_true",
        help="Print language identifier of the message catalog source file",
    )
    parser.add_argument(
        "--prefix",
        default="kmp_i18n",
        help="Prefix to be used for all C identifiers (type and variable names)"
        " in enum and default message files.",
    )
    parser.add_argument("--enum", metavar="FILE", help="Generate enum file named FILE")
    parser.add_argument(
        "--default", metavar="FILE", help="Generate default messages file named FILE"
    )
    parser.add_argument(
        "--signature", metavar="FILE", help="Generate signature file named FILE"
    )
    parser.add_argument(
        "--message", metavar="FILE", help="Generate message file named FILE"
    )
    parser.add_argument(
        "--target-system-override",
        metavar="TARGET_SYSTEM_NAME",
        help="Target System override.\n"
        "By default the target system is the host system\n"
        "See possible values at https://docs.python.org/3/library/platform.html#platform.system",
    )
    parser.add_argument("inputfile")
    commandArgs = parser.parse_args()

    if commandArgs.lang_id:
        display_language_id(commandArgs.inputfile)
        return
    data = MessageData.create(commandArgs.inputfile)
    prefix = commandArgs.prefix
    if commandArgs.target_system_override:
        TargetPlatform.set_system_override(commandArgs.target_system_override)
    if commandArgs.enum:
        generate_enum_file(commandArgs.enum, prefix, data)
    if commandArgs.default:
        generate_default_messages_file(commandArgs.default, prefix, data)
    if commandArgs.signature:
        generate_signature_file(commandArgs.signature, data)
    if commandArgs.message:
        if TargetPlatform.system().casefold() == "Windows".casefold():
            generate_message_file_windows(commandArgs.message, data)
        else:
            generate_message_file_unix(commandArgs.message, data)


if __name__ == "__main__":
    try:
        main()
    except ScriptError as e:
        print("error: {}".format(e))
        sys.exit(1)

# end of file
