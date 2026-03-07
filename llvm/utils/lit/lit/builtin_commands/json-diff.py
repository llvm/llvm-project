import json
import sys
import difflib
import getopt
import re

# ==============================================================================
# Utility Functions
# ==============================================================================


def fail(exit_code, *messages):
    """Print error messages to stderr and exit with the given code."""
    for msg in messages:
        sys.stderr.write(msg)
        sys.stderr.write("\n")
    sys.exit(exit_code)


# ==============================================================================
# Match Result Classes
# ==============================================================================


class MatchResult:
    """Result of a pattern match operation, with success/failure state."""

    def __init__(self, message):
        self.message = message

    def __bool__(self):
        return self.message is None

    @classmethod
    def success(cls):
        """Create a successful match result."""
        return cls(None)

    @classmethod
    def failure(cls, message):
        """Create a failed match result with an error message."""
        return cls(message)


class MatchContext:
    """Maintains state during JSON comparison traversal."""

    def __init__(self, captures, definitions, ignore_extra_keys):
        self.captures = captures
        self.definitions = definitions
        self.ignore_extra_keys = ignore_extra_keys


# ==============================================================================
# Pattern Classes
# ==============================================================================


class Pattern:
    """Base class for all pattern types."""

    def match(self, actual_value, match_context):
        """Match pattern against actual value, returning MatchResult."""
        fail(2, "Error: called match method on base Pattern class.")


class AbsentValue:
    """Sentinel value representing an absent key in actual JSON."""

    pass


class AbsencePattern(Pattern):
    """@! - Absence assertion, key must not exist in actual JSON."""

    def match(self, actual_value, match_context):
        """Check that the key is absent in actual JSON."""
        if isinstance(actual_value, AbsentValue):
            return MatchResult.success()
        else:
            message = f"Match failure: pattern '@!' requires the key to be absent, got '{actual_value}'."
            return MatchResult.failure(message)


class RegexPattern(Pattern):
    """@#pattern - Regex matching for string values."""

    def __init__(self, regex_pattern):
        try:
            self.regex = re.compile(regex_pattern)
        except re.error as e:
            fail(2, f"Error: invalid regex pattern '{regex_pattern}': {e}.")

    def match(self, actual_value, match_context):
        """Check if actual value matches the regex pattern."""
        if not isinstance(actual_value, str):
            message = f"Match failure: regex pattern '{self.regex.pattern}' requires string, got '{type(actual_value).__name__}'."
            return MatchResult.failure(message)
        elif self.regex.search(actual_value):
            return MatchResult.success()
        else:
            message = f"Match failure: value '{actual_value}' does not match regex pattern '{self.regex.pattern}'."
            return MatchResult.failure(message)


class TemplatePattern(Pattern):
    """@%template - Template interpolation with command-line values."""

    def __init__(self, template):
        self.template = template

    def match(self, actual_value, match_context):
        """Check if actual value matches the interpolated template."""
        try:
            interpolated = self.template.format(**match_context.definitions)
            if actual_value == interpolated:
                return MatchResult.success()
            else:
                message = f"Match failure: value '{actual_value}' does not match template '{self.template}' expanded to '{interpolated}'."
                return MatchResult.failure(message)
        except KeyError as e:
            message = (
                f"Error: template '{self.template}' references undefined variable: {e}."
            )
            fail(2, message)
        except Exception as e:
            fail(2, f"Error: template '{self.template}' expansion failed: {e}.")


class CapturePattern(Pattern):
    """@&NAME - Capture value and ensure consistency across all occurrences."""

    def __init__(self, name):
        self.name = name

    def match(self, actual_value, match_context):
        """Capture value on first occurrence, check consistency on subsequent ones."""
        if self.name not in match_context.captures:
            match_context.captures[self.name] = actual_value
            return MatchResult.success()
        else:
            captured = match_context.captures[self.name]
            if actual_value == captured:
                return MatchResult.success()
            else:
                message = f"Match failure: capture '{self.name}' mismatch: expected '{captured}', got '{actual_value}'."
                return MatchResult.failure(message)


# ==============================================================================
# Pattern Recognition and Parsing
# ==============================================================================


def recognizePattern(value_str):
    """
    Parse a string value and return Pattern object if it matches a pattern.
    """

    assert isinstance(value_str, str)

    # Order matters: @@ must be checked before other @ patterns
    if value_str.startswith("@@"):
        return value_str[2:]

    elif value_str == "@!":
        return AbsencePattern()

    elif value_str.startswith("@&"):
        name = value_str[2:]
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            fail(2, f"Error: invalid capture name: '{name}'.")
        return CapturePattern(name)

    elif value_str.startswith("@#"):
        pattern = value_str[2:]
        return RegexPattern(pattern)

    elif value_str.startswith("@%"):
        template = value_str[2:]
        return TemplatePattern(template)

    else:
        return value_str


def parsePatterns(obj):
    """
    Recursively traverse JSON object and replace pattern strings with Pattern objects.
    """

    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            result[key] = parsePatterns(value)
        return result

    elif isinstance(obj, list):
        return [parsePatterns(element) for element in obj]

    elif isinstance(obj, str):
        return recognizePattern(obj)

    else:
        return obj


# ==============================================================================
# Pattern Matching
# ==============================================================================


def patternMatchJson(expected, actual, match_context):
    """
    Recursively compare expected and actual JSON with pattern support.
    """

    if isinstance(expected, Pattern):
        result = expected.match(actual, match_context)
        if not result:
            return result

    elif type(expected) is not type(actual):
        message = f"Match failure: type mismatch: expected '{type(expected).__name__}', got '{type(actual).__name__}'."
        return MatchResult.failure(message)

    elif isinstance(expected, dict):
        return patternMatchDicts(expected, actual, match_context)

    elif isinstance(expected, list):
        return patternMatchLists(expected, actual, match_context)

    elif expected != actual:
        message = (
            f"Match failure: value mismatch: expected '{expected}', got '{actual}'."
        )
        return MatchResult.failure(message)

    return MatchResult.success()


def patternMatchDicts(expected, actual, match_context):
    """Compare two dictionaries with pattern support."""

    expected_keys = set(expected.keys())
    actual_keys = set(actual.keys())

    for key in expected.keys():
        expected_value = expected[key]
        actual_value = actual.get(key, AbsentValue())
        match_result = patternMatchJson(expected_value, actual_value, match_context)
        if not match_result:
            return match_result

    extra_actual_keys = actual_keys - expected_keys

    if extra_actual_keys and not match_context.ignore_extra_keys:
        message = f"Match failure: extra keys in actual JSON: '{extra_actual_keys}'."
        return MatchResult.failure(message)

    return MatchResult.success()


def patternMatchLists(expected, actual, match_context):
    """Compare two lists with pattern support."""

    if len(expected) != len(actual):
        message = f"Match failure: array length mismatch: expected '{len(expected)}', got '{len(actual)}'."
        return MatchResult.failure(message)

    for expected_value, actual_value in zip(expected, actual):
        match_result = patternMatchJson(expected_value, actual_value, match_context)
        if not match_result:
            return match_result

    return MatchResult.success()


# ==============================================================================
# Diff Generation
# ==============================================================================


def normalizeJson(obj, indent=2):
    """Serialize JSON object to a canonical string representation."""
    return json.dumps(obj, indent=indent, sort_keys=True, ensure_ascii=False)


def diffJson(expected, actual, expected_file, actual_file, context_lines):
    """
    Generate colored diff with optional pattern error details.
    """

    expected_lines = normalizeJson(expected).splitlines(keepends=True)
    actual_lines = normalizeJson(actual).splitlines(keepends=True)

    diff = difflib.unified_diff(
        expected_lines,
        actual_lines,
        fromfile=expected_file,
        tofile=actual_file,
        n=context_lines,
        lineterm="",
    )

    green = "\x1b[0;32m"
    red = "\x1b[0;31m"
    normal = "\x1b[0m"

    colored_diff = []

    for line in diff:
        if line.startswith("+ "):
            colored_diff.append(f"{green}{line}{normal}")
        elif line.startswith("- "):
            colored_diff.append(f"{red}{line}{normal}")
        else:
            colored_diff.append(line)

    return "".join(colored_diff)


# ==============================================================================
# Comparison Entry Point
# ==============================================================================


def compareJson(actual, expected, args):
    """Main comparison entry point with pattern support."""

    expected_with_patterns = parsePatterns(expected)

    match_context = MatchContext({}, args.definitions, args.ignore_extra_keys)

    match_result = patternMatchJson(expected_with_patterns, actual, match_context)

    if not match_result:
        diff = diffJson(
            expected, actual, args.expected_file, args.actual_file, args.context
        )
        fail(1, match_result.message, diff)


# ==============================================================================
# Command-line Parsing
# ==============================================================================


class CommandLineArguments:
    """Container for parsed command-line arguments."""

    def __init__(self):
        self.ignore_extra_keys = False
        self.context = 3
        self.expected_file = None
        self.actual_file = None
        self.definitions = {}


def parseCommandLine():
    """Parse command-line arguments and return CommandLineArguments object."""
    try:
        opts, args = getopt.gnu_getopt(
            sys.argv[1:], "ic:", ["ignore-extra-keys", "context=", "define="]
        )
    except getopt.GetoptError as err:
        fail(2, f"Error: failed to parse command-line arguments: {err}.")

    flags = CommandLineArguments()

    for opt, arg in opts:
        if opt in ("-i", "--ignore-extra-keys"):
            flags.ignore_extra_keys = True
        elif opt in ("-c", "--context"):
            try:
                flags.context = int(arg)
                if flags.context < 0:
                    raise ValueError()
            except ValueError:
                fail(2, f"Error: invalid context value: '{arg}'.")
        elif opt == "--define":
            if "=" not in arg:
                message = (
                    f"Error: invalid --define format: '{arg}'. Expected name=value."
                )
                fail(2, message)
            name, value = arg.split("=", 1)
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
                fail(2, f"Error: invalid variable name in --define: '{name}'.")
            flags.definitions[name] = value

    if len(args) != 2:
        fail(2, "Error: expected two positional arguments.")

    flags.expected_file = args[0]
    flags.actual_file = args[1]

    return flags


# ==============================================================================
# JSON I/O
# ==============================================================================


def readJson(filepath):
    """Read and parse JSON file, checking for duplicate keys."""

    def checkDuplicateKeys(pairs):
        """Check for duplicate keys in JSON object during parsing."""
        keys = [key for (key, value) in pairs]
        seen = set()
        duplicates = set()

        for key in keys:
            if key in seen:
                duplicates.add(key)
            else:
                seen.add(key)

        if duplicates:
            dupkeys = ", ".join(sorted(duplicates))
            fail(2, f"Error: failed to read JSON. Found duplicate keys: {dupkeys}.")

        return dict(pairs)

    try:
        with open(filepath, "r") as f:
            data = json.load(f, object_pairs_hook=checkDuplicateKeys)
            return data
    except Exception as e:
        fail(2, f"Error: failed to read JSON from '{filepath}': {e}.")


# ==============================================================================
# Main Entry Point
# ==============================================================================


def main():
    """Main entry point for json-diff command."""
    args = parseCommandLine()

    expected = readJson(args.expected_file)
    actual = readJson(args.actual_file)

    compareJson(actual, expected, args)


if __name__ == "__main__":
    main()
