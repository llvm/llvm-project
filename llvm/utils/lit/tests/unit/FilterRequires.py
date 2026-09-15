# RUN: %{python} %s

import pickle
import unittest
from unittest.mock import patch

from lit.FilterRequires import FilterRequires


class TestFilterRequires(unittest.TestCase):
    def test_selection(self):
        cases = [
            ("Half", "Half", True),
            ("Int16", "Half", False),
            ("Half && Int16", "Half", False),
            ("Half", "Half && Int16", False),
            ("Int16", "Half && Int16", False),
            ("Half && Int16", "Half && Int16", True),
            ("Half && Int16 && Double", "Half && Int16", False),
            ("Half || Int16", "Half && Int16", False),
            ("Half && Int16", "Half || Int16", False),
            ("Half || Int16", "Half || Int16", True),
            ("Half", "Half || Int16", True),
            ("Int16", "Half || Int16", True),
            ("Half", "Half && !Double", True),
            ("Half && !Double", "Half && !Double", True),
            ("Half && Double", "Half && !Double", False),
            ("Half && !Double", "Half", False),
            ("!Double", "!Double", True),
            ("Double", "!Double", False),
            ("Half", "Half && !Vulkan", True),
            ("Half", "Half && !DirectX && !Vulkan", True),
            ("Half && (Int16 || Double)", "Half && Int16 && !Double", True),
            (
                "Int64 && SM_6_6 && (!Vulkan || VulkanInt64BufferAtomics)",
                "Int64 && SM_6_6 && !Vulkan",
                True,
            ),
            (
                "Int64 && SM_6_6 && (!Vulkan || VulkanInt64BufferAtomics)",
                "Int64 && SM_6_6",
                False,
            ),
            ("!DirectX || ResourceBindingTier3", "!Vulkan", False),
            (
                "!DirectX || ResourceBindingTier3",
                "ResourceBindingTier3 && !Vulkan",
                True,
            ),
            ("Half && !DirectX", "Half && !Vulkan", False),
            ("Half", "Half && !Half", False),
            ("Half && !Half", "Half && !Half", False),
            ("Half || (Double && !Double)", "Half", True),
            ("Half && (Int16 || !Int16)", "Half", False),
            ("Half || (Half && Int16)", "Half && Int16", True),
            ("Half", "Half || (Half && Int16)", True),
            ("!(Half || Double)", "!Half && !Double", True),
            ("!(Half && Double)", "!Half", True),
            ("!(Half && Double)", "true", False),
            ("!(!Half || Double)", "Half && !Double", True),
            ("!!Half", "Half", True),
            ("Half", "!!Half", True),
            ("!(Half && !Int16)", "Int16", True),
            (" Half && (Int16) ", " Int16, Half, Half ", True),
            ("Half, Int16", "Int16 && Half", True),
            ("half", "Half", False),
            ("HalfExtra", "Half", False),
            ("target=x86_64-pc-windows", "target=x86_64-pc-windows", True),
            ("true && Half", "Half", True),
            ("true", "true", True),
            ("true && true", "true", True),
            ("!!true", "true", True),
            ("true || Half", "true", True),
            ("true || Half", "Half", True),
            ("!true", "true", False),
            ("!true", "!true", False),
            ("Half", "true", False),
            ("!Double", "true", False),
            ("true", "Half", False),
            ("true", "!Double", True),
            ("true", "True", False),
            ("false", "false", True),
            ("Base", "Base", False),
            ("Base && Half", "Base && Half", True),
        ]
        for requirement, selection, expected in cases:
            with self.subTest(requirement=requirement, selection=selection):
                self.assertEqual(
                    FilterRequires(selection).matches([requirement]), expected
                )

    def test_base_and_multiple_entries(self):
        self.assertTrue(FilterRequires(" Base ").matches([]))
        self.assertFalse(FilterRequires("Base").matches(["true"]))
        self.assertFalse(FilterRequires("true").matches([]))
        self.assertFalse(FilterRequires("Half || true").matches([]))
        self.assertTrue(
            FilterRequires("Int16 && Half").matches(["Half", "true", "Int16"])
        )
        self.assertFalse(FilterRequires("Half").matches(["Half", "!Double"]))
        self.assertTrue(FilterRequires("Half && !Double").matches(["Half", "!Double"]))

    def test_errors(self):
        for expression in [
            "",
            " ",
            "Half &&",
            "Half Int16",
            "()",
            "(Half",
            "Half)",
            "Half & Int16",
            "Half | Int16",
            "Half,",
            ",Half",
            "Half,,Int16",
            "(Half,Int16)",
            "*",
            "Half*",
            "Half?",
            "Half[0]",
            "{{.*}}",
            "target={{.*}}",
            "Half || {{.*}}",
        ]:
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    FilterRequires(expression)
                with self.assertRaises(ValueError):
                    FilterRequires("Half").matches([expression])

    def test_limits(self):
        # Even if an early combination matches, never silently truncate the rest.
        expression = "Half || " + " || ".join("f%d" % i for i in range(1024))
        with self.assertRaisesRegex(ValueError, "combination limit"):
            FilterRequires(expression)
        with self.assertRaisesRegex(ValueError, "combination limit"):
            FilterRequires("Half").matches([expression])
        with patch.object(FilterRequires, "MAX_STEPS", 100):
            with self.assertRaisesRegex(ValueError, "normalization limit"):
                FilterRequires(" && ".join(["(a || b)"] * 20))
            with self.assertRaisesRegex(ValueError, "normalization limit"):
                FilterRequires("Half").matches([" && ".join(["(a || b)"] * 20)])
        with self.assertRaisesRegex(ValueError, "nested too deeply"):
            FilterRequires("(" * 2000 + "Half" + ")" * 2000)

    def test_cartesian_limit(self):
        expression = " && ".join("(a%d || b%d)" % (i, i) for i in range(11))
        with self.assertRaisesRegex(ValueError, "combination limit"):
            FilterRequires(expression)
        # Deduplication keeps repeated, rather than independent, choices small.
        self.assertTrue(FilterRequires(" && ".join(["(a || b)"] * 20)).matches(["a"]))

    def test_pickling(self):
        selection = pickle.loads(pickle.dumps(FilterRequires("Half && !Double")))
        self.assertTrue(selection.matches(["Half"]))
        self.assertFalse(selection.matches(["Half && Double"]))
        self.assertTrue(pickle.loads(pickle.dumps(FilterRequires("Base"))).matches([]))


if __name__ == "__main__":
    unittest.main()
